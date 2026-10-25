use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Emitter, Manager, State, Window};

// 模型运行状态
#[derive(Default)]
enum ModelState {
    #[default]
    Loading,
    Ready,
    Error(String),
}

struct ModelStatus {
    state: Mutex<ModelState>,
}

// 读取用户选择的图片文件，返回原始字节
#[tauri::command]
fn read_image(path: String) -> Result<Vec<u8>, String> {
    fs::read(&path).map_err(|e| format!("读取图片失败: {e}"))
}

// 把 PNG 字节保存到用户指定的路径
#[tauri::command]
fn save_png(bytes: Vec<u8>, path: String) -> Result<String, String> {
    // 校验确实是 PNG
    image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
        .map_err(|e| format!("不是有效的 PNG 数据: {e}"))?;

    let dest = PathBuf::from(&path);
    let parent = dest
        .parent()
        .ok_or_else(|| "无效的保存路径".to_string())?;
    fs::create_dir_all(parent).map_err(|e| format!("无法创建目录: {e}"))?;
    fs::write(&dest, &bytes).map_err(|e| format!("保存失败: {e}"))?;
    Ok(dest.to_string_lossy().into_owned())
}

// ---------------- 推理 ----------------

const MODEL_INPUT: (usize, usize) = (1024, 1024);
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

struct InferenceState {
    session: Mutex<Option<Session>>,
}

// 解码图片 -> resize 1024x1024 -> ImageNet 归一化 -> NCHW tensor
fn preprocess(img: &DynamicImage) -> Array4<f32> {
    let resized = img
        .resize_exact(
            MODEL_INPUT.0 as u32,
            MODEL_INPUT.1 as u32,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();

    let mut arr = Array4::<f32>::zeros((1, 3, MODEL_INPUT.1, MODEL_INPUT.0));
    for (x, y, px) in resized.enumerate_pixels() {
        for c in 0..3 {
            arr[[0, c, y as usize, x as usize]] =
                (px[c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    arr
}

// 跑一次推理，返回 0~1 概率图（1024x1024）
fn run_inference(
    state: &InferenceState,
    arr: Array4<f32>,
) -> Result<(usize, usize, Vec<f32>), String> {
    let mut guard = state
        .session
        .lock()
        .map_err(|_| "推理会话被占用".to_string())?;
    let session = guard
        .as_mut()
        .ok_or_else(|| "模型尚未加载完成，请稍候或重新加载".to_string())?;

    let input = Tensor::from_array(arr).map_err(|e| format!("构造 tensor 失败: {e}"))?;
    let outputs = session
        .run(ort::inputs!["input_image" => input])
        .map_err(|e| format!("推理失败: {e}"))?;

    let output = outputs["output_image"]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("读取输出失败: {e}"))?;
    let (shape, data) = output;
    let dims = shape.to_vec();
    let (map_h, map_w) = (dims[2] as usize, dims[3] as usize);
    // 该导出输出的是 logits，需要 Sigmoid 转成 0~1 概率
    let prob: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    Ok((map_h, map_w, prob))
}

// ---------------- 硬边后处理 ----------------

// 3x3 腐蚀：输出像素为 255 当且仅当 3x3 邻域内全部为 255（前景）
fn erode_3x3(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut fg = 255u8;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    // 越界视为背景
                    let v = if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        0
                    } else {
                        src[ny as usize * w + nx as usize]
                    };
                    fg = fg.min(v);
                }
            }
            dst[y * w + x] = fg;
        }
    }
    dst
}

// 3x3 膨胀：输出像素为 255 只要 3x3 邻域内有任一 255
fn dilate_3x3(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut dst = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut fg = 0u8;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let v = src[ny as usize * w + nx as usize];
                    fg = fg.max(v);
                }
            }
            dst[y * w + x] = fg;
        }
    }
    dst
}

// 对图片做主体分割，返回 RGBA PNG
#[tauri::command]
fn segment_image(
    state: State<InferenceState>,
    window: Window,
    image_bytes: Vec<u8>,
    hard_edge: bool,
    smooth: bool,
) -> Result<Vec<u8>, String> {
    segment_image_core(&state, Some(&window), &image_bytes, hard_edge, smooth)
}

fn segment_image_core(
    state: &InferenceState,
    window: Option<&Window>,
    image_bytes: &[u8],
    hard_edge: bool,
    smooth: bool,
) -> Result<Vec<u8>, String> {
    if let Some(w) = window {
        let _ = w.emit("pipeline-stage", "inference");
    }
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| format!("图片解码失败，可能是格式不支持: {e}"))?;
    let (orig_w, orig_h) = (img.width(), img.height());

    let input_arr = preprocess(&img);
    let (map_h, map_w, map) = run_inference(&state, input_arr)?;

    if let Some(w) = window {
        let _ = w.emit("pipeline-stage", "postprocess");
    }

    // 概率图 -> 灰度图 -> 缩放回原图尺寸，得到软 alpha
    let mut gray = image::GrayImage::new(map_w as u32, map_h as u32);
    for (x, y, px) in gray.enumerate_pixels_mut() {
        let v = map[y as usize * map_w + x as usize].clamp(0.0, 1.0);
        *px = image::Luma([(v * 255.0).round() as u8]);
    }
    let scaled = image::imageops::resize(
        &gray,
        orig_w,
        orig_h,
        image::imageops::FilterType::Lanczos3,
    );

    // 最终 alpha 缓冲（0~255）
    let mut alpha: Vec<u8> = scaled.into_raw();

    if hard_edge {
        // 二值硬边 + 可选 1px 羽化
        for a in alpha.iter_mut() {
            *a = if *a >= 128 { 255 } else { 0 };
        }
        let eroded = erode_3x3(&alpha, orig_w as usize, orig_h as usize);
        alpha = dilate_3x3(&eroded, orig_w as usize, orig_h as usize);
        for a in alpha.iter_mut() {
            *a = if *a >= 128 { 255 } else { 0 };
        }
        if smooth {
            if let Some(gray) = image::GrayImage::from_raw(orig_w, orig_h, alpha.clone()) {
                alpha = image::imageops::blur(&gray, 1.0).into_raw();
            }
        }
    } else {
        // 平滑软边（默认）：先去噪（开运算去除碎片毛边），再 1~1.5px 高斯羽化
        // 形成抗锯齿的平滑过渡边缘，接近 PS 钢笔抠图的干净轮廓
        let mut bin = alpha.clone();
        for a in bin.iter_mut() {
            *a = if *a >= 128 { 255 } else { 0 };
        }
        let eroded = erode_3x3(&bin, orig_w as usize, orig_h as usize);
        let opened = dilate_3x3(&eroded, orig_w as usize, orig_h as usize);
        if let Some(gray) = image::GrayImage::from_raw(orig_w, orig_h, opened.clone()) {
            alpha = image::imageops::blur(&gray, 1.2).into_raw();
        } else {
            alpha = opened;
        }
    }

    // 4. 与原图 RGB 合成 RGBA PNG
    let rgb = img.to_rgb8();
    let mut rgba = image::RgbaImage::new(orig_w, orig_h);
    for (x, y, px) in rgba.enumerate_pixels_mut() {
        let i = (y as usize) * (orig_w as usize) + x as usize;
        let c = rgb.get_pixel(x, y);
        *px = image::Rgba([c[0], c[1], c[2], alpha[i]]);
    }

    if let Some(w) = window {
        let _ = w.emit("pipeline-stage", "encoding");
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(rgba)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| format!("PNG 编码失败: {e}"))?;
    if let Some(w) = window {
        let _ = w.emit("pipeline-stage", "done");
    }
    Ok(png)
}

// ---------------- 魔法橡皮擦（需求 9） ----------------

// 从点击点出发，把"连在一起且颜色相近"的前景像素全部清除（alpha 置 0）
// 仅沿着前景（alpha > 0）蔓延，不会穿越已经透明的洞
#[tauri::command]
fn erase_region(image_bytes: Vec<u8>, x: i32, y: i32, tolerance: i32) -> Result<Vec<u8>, String> {
    let mut img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("图片解码失败: {e}"))?
        .to_rgba8();
    let (w, h) = (img.width() as i32, img.height() as i32);
    if x < 0 || y < 0 || x >= w || y >= h {
        return Err("点击位置超出图片范围".to_string());
    }

    let seed = *img.get_pixel(x as u32, y as u32);
    if seed[3] == 0 {
        return Err("点击的位置已经是透明区域，请点击要清除的前景部分".to_string());
    }
    let (sr, sg, sb) = (seed[0] as i32, seed[1] as i32, seed[2] as i32);
    let tol = tolerance.clamp(0, 255);

    // 洪水填充（迭代式 DFS，避免大图递归爆栈）
    let mut visited = vec![false; (w * h) as usize];
    let mut stack: Vec<(i32, i32)> = vec![(x, y)];
    visited[(y * w + x) as usize] = true;

    while let Some((cx, cy)) = stack.pop() {
        img.get_pixel_mut(cx as u32, cy as u32)[3] = 0; // 清除
        for (nx, ny) in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)] {
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            let idx = (ny * w + nx) as usize;
            if visited[idx] {
                continue;
            }
            let p = img.get_pixel(nx as u32, ny as u32);
            let similar = (p[0] as i32 - sr).abs() <= tol
                && (p[1] as i32 - sg).abs() <= tol
                && (p[2] as i32 - sb).abs() <= tol;
            if p[3] > 0 && similar {
                visited[idx] = true;
                stack.push((nx, ny));
            }
        }
    }

    let mut png = Vec::new();
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| format!("PNG 编码失败: {e}"))?;
    Ok(png)
}

// 列出文件夹中受支持的图片文件名
#[tauri::command]
fn list_image_files(dir: String) -> Result<Vec<String>, String> {
    let supported = ["png", "jpg", "jpeg", "webp", "bmp"];
    let mut names: Vec<String> = fs::read_dir(&dir)
        .map_err(|e| format!("读取文件夹失败: {e}"))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let ext = name.rsplit('.').next()?.to_lowercase();
            if supported.contains(&ext.as_str()) {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    names.sort();
    Ok(names)
}

// 查询模型状态
#[tauri::command]
fn get_model_status(status: State<ModelStatus>) -> Result<String, String> {
    let st = status.state.lock().map_err(|_| "状态读取失败".to_string())?;
    Ok(match &*st {
        ModelState::Loading => "loading".to_string(),
        ModelState::Ready => "ready".to_string(),
        ModelState::Error(msg) => format!("error: {msg}"),
    })
}

// 加载或重载模型
fn load_model(app: &AppHandle, status: &ModelStatus) -> Result<(), String> {
    {
        let mut st = status.state.lock().map_err(|_| "状态锁失败".to_string())?;
        *st = ModelState::Loading;
    }
    let _ = app.emit("model-status", "loading");

    let result = (|| {
        let model_path = resolve_model_path(app)?;
        let environment = ort::init()
            .with_name("koutu")
            .build()
            .map_err(|e| e.to_string())?;
        Session::builder(&environment)
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .with_intra_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get().min(8))
                    .unwrap_or(4),
            )
            .map_err(|e| e.to_string())?
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
            .map_err(|e| e.to_string())?
            .commit_from_file(&model_path)
            .map_err(|e| format!("模型加载失败: {e}"))
    })();

    let mut st = status.state.lock().map_err(|_| "状态锁失败".to_string())?;
    match result {
        Ok(session) => {
            let inference = app.state::<InferenceState>();
            let mut slot = inference
                .session
                .lock()
                .map_err(|_| "推理会话锁失败".to_string())?;
            *slot = Some(session);
            *st = ModelState::Ready;
            let _ = app.emit("model-status", "ready");
        }
        Err(e) => {
            *st = ModelState::Error(e.clone());
            let _ = app.emit("model-status", format!("error: {e}"));
        }
    }
    Ok(())
}

// 重新加载模型（加载失败后重试）
#[tauri::command]
fn reload_model(app: AppHandle, status: State<ModelStatus>) -> Result<(), String> {
    load_model(&app, &status)
}

fn resolve_model_path(app: &AppHandle) -> Result<PathBuf, String> {
    // 1. 安装包内：BaseDirectory::Resource
    if let Ok(p) = app
        .path()
        .resolve("models/BiRefNet.onnx", BaseDirectory::Resource)
    {
        if p.exists() {
            return Ok(p);
        }
    }
    // 2. dev 环境回退：相对路径
    let candidates = [
        PathBuf::from("models/BiRefNet.onnx"),
        PathBuf::from("src-tauri/models/BiRefNet.onnx"),
    ];
    for c in &candidates {
        if c.exists() {
            return Ok(c.clone());
        }
    }
    // 3. 便携版：可执行文件同级目录下的 models/
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let p = parent.join("models/BiRefNet.onnx");
            if p.exists() {
                return Ok(p);
            }
        }
    }
    Err(format!(
        "找不到模型文件，请确认安装包内包含 models/BiRefNet.onnx"
    ))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            app.manage(ModelStatus { state: Mutex::new(ModelState::Loading) });
            app.manage(InferenceState { session: Mutex::new(None) });
            let status = app.state::<ModelStatus>();
            load_model(app.handle(), &status)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            read_image,
            save_png,
            segment_image,
            erase_region,
            get_model_status,
            reload_model,
            list_image_files
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Instant;

    fn make_test_session() -> Session {
        let model_path = std::path::PathBuf::from("src-tauri/models/BiRefNet.onnx");
        assert!(model_path.exists(), "找不到模型");
        let env = ort::init().with_name("test").build().unwrap();
        Session::builder(&env)
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap()
            .with_intra_threads(2)
            .unwrap()
            .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
            .unwrap()
            .commit_from_file(&model_path)
            .unwrap()
    }

    fn make_sample_png() -> Vec<u8> {
        // 800x600：深色背景 + 中间白色矩形（模拟"产品主体"）
        let mut img = image::RgbImage::from_pixel(800, 600, image::Rgb([30, 60, 40]));
        for y in 150..450 {
            for x in 250..550 {
                img.put_pixel(x, y, image::Rgb([245, 245, 245]));
            }
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    #[test]
    fn test_hard_edge_binary_alpha() {
        let session = make_test_session();
        let state = InferenceState { session: Mutex::new(Some(session)) };
        let img_bytes = make_sample_png();

        // ---- 硬边模式 ----
        let t0 = Instant::now();
        let png = segment_image_core(&state, None, &img_bytes, true, false).unwrap();
        println!("硬边推理+后处理耗时: {:.2?}", t0.elapsed());

        let out = image::load_from_memory(&png).unwrap().to_rgba8();
        let (w, h) = (out.width() as usize, out.height() as usize);
        assert_eq!((w, h), (800, 600));

        // 验收 1：alpha 直方图只有 0 和 255
        let mut hist = BTreeMap::new();
        for p in out.pixels() {
            *hist.entry(p[3]).or_insert(0usize) += 1;
        }
        let keys: Vec<u8> = hist.keys().copied().collect();
        println!("硬边 alpha 直方图: {:?}", hist);
        assert!(
            keys.iter().all(|k| *k == 0 || *k == 255),
            "alpha 应只有 0/255，实际出现: {:?}",
            keys
        );
        assert!(hist.contains_key(&255) && hist.contains_key(&0), "应同时存在 0 和 255");

        // 验收 2：合成正确——中心应为白色且不透明，角落应为背景色且透明
        let center = out.get_pixel(400, 300);
        let corner = out.get_pixel(10, 10);
        println!("中心 RGBA: {:?}, 角落 RGBA: {:?}", center, corner);
        assert_eq!(center, &image::Rgba([245, 245, 245, 255]));
        assert_eq!(corner, &image::Rgba([30, 60, 40, 0]));

        let _ = out.save(std::env::temp_dir().join("test_hard_edge.png"));
        println!("硬边 PNG 已保存(临时目录)");

        // ---- 软边模式（对比）----
        let png_soft = segment_image_core(&state, None, &img_bytes, false, false).unwrap();
        let out_soft = image::load_from_memory(&png_soft).unwrap().to_rgba8();
        let mut soft_hist = BTreeMap::new();
        for p in out_soft.pixels() {
            *soft_hist.entry(p[3]).or_insert(0usize) += 1;
        }
        let soft_keys: Vec<u8> = soft_hist.keys().copied().collect();
        println!("软边 alpha 取值数: {}, 样例: {:?}", soft_keys.len(), &soft_keys[..soft_keys.len().min(8)]);
        assert!(
            soft_keys.iter().any(|k| *k != 0 && *k != 255),
            "软边应存在中间值，否则对比测试无意义"
        );
        let _ = out_soft.save(std::env::temp_dir().join("test_soft_edge.png"));
        println!("软边 PNG 已保存(临时目录)");
    }
}

#[cfg(test)]
mod erase_tests {
    use super::*;

    fn make_result_png() -> Vec<u8> {
        // 200x100：左半红色方块（不透明），右半透明
        let mut img = image::RgbaImage::new(200, 100);
        for y in 0..100 {
            for x in 0..200 {
                if x < 100 {
                    img.put_pixel(x, y, image::Rgba([220, 30, 30, 255]));
                } else {
                    img.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
                }
            }
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    #[test]
    fn test_erase_region_flood_fill() {
        let png = make_result_png();
        let out = erase_region(png, 50, 50, 30).unwrap();
        let img = image::load_from_memory(&out).unwrap().to_rgba8();
        // 整个红色连通块都应被清除
        assert_eq!(img.get_pixel(50, 50)[3], 0);
        assert_eq!(img.get_pixel(0, 0)[3], 0);
        assert_eq!(img.get_pixel(99, 99)[3], 0);
        // 右半本来透明，不受影响
        assert_eq!(img.get_pixel(150, 50)[3], 0);
    }

    #[test]
    fn test_erase_region_rejects_transparent_click() {
        let png = make_result_png();
        let err = erase_region(png, 150, 50, 30);
        assert!(err.is_err(), "点在透明区应报错");
    }
}

#[cfg(test)]
mod list_tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_list_image_files() {
        let dir = std::env::temp_dir().join("koutu_list_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("b.jpg"), b"x").unwrap();
        fs::write(dir.join("a.PNG"), b"x").unwrap();
        fs::write(dir.join("notes.txt"), b"x").unwrap();
        let names = list_image_files(dir.to_string_lossy().into_owned()).unwrap();
        assert_eq!(names, vec!["a.PNG".to_string(), "b.jpg".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }
}
