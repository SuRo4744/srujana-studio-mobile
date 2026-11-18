import * as THREE from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as GaussianSplats3D from "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.6/build/gaussian-splats-3d.module.js";

const $ = id => document.getElementById(id);
const stages = ["queued", "filtering", "reconstructing", "done"];
const settingIds = ["pointSizeRange", "opacityRange", "brightnessRange", "finishRange", "rotateX", "rotateY", "rotateZ", "scaleRange", "adaptiveClarity", "renderStyle"];
const defaultSettings = {
  pointSizeRange: "3", opacityRange: "1", brightnessRange: "1", finishRange: "35",
  rotateX: "0", rotateY: "0", rotateZ: "0", scaleRange: "1",
  adaptiveClarity: true, renderStyle: "points"
};
let state = {kind: null, splat: null, renderer: null, scene: null, camera: null, controls: null,
  group: null, pointMesh: null, surfaceMesh: null, pointsMaterial: null, meshMaterial: null,
  originalGeometry: null, surfaceGeometry: null, raf: null, originalRotation: 0};
let currentJob = null;
let originalVideoUrl = null;
let autoRotate = false;
let darkCanvas = true;
let pollTimer = null;
let cloudMaxUploadMB = null;
async function configureDeployment() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) return;
    const config = await response.json();
    cloudMaxUploadMB = config.max_upload_mb;
    if (config.cloud) {
      $("cloudNotice").hidden = false;
      for (const option of Array.from($("mode").options)) if (option.value !== "turntable") option.remove();
      $("mode").value = "turntable";
      for (const option of Array.from($("qualityPreset").options)) {
        if (option.value === "high") option.remove();
        if (option.value === "fast") option.textContent = "Fast — cloud optimized";
        if (option.value === "balanced") option.textContent = "Balanced — cloud optimized";
      }
      $("qualityPreset").value = "fast";
      $("dlBlend").hidden = true;
      document.querySelector('label[for="recentJobs"]').textContent = "Recent jobs (temporary cloud storage)";
      $("downloadGeometry").closest(".export-preference").querySelector("label").textContent = "OBJ geometry";
    }
  } catch (_) { /* The app still works locally if configuration is unavailable. */ }
}
configureDeployment();

function message(text, isError = false) {
  const box = $("msg");
  if (!text) { box.style.display = "none"; return; }
  box.textContent = text;
  box.className = "show" + (isError ? " err" : "");
  box.style.display = "block";
}
function apiError(response, detail) {
  return new Error(detail?.detail || `Request failed (${response.status})`);
}
function updateStep(stage, msg = "") {
  const currentIndex = stages.indexOf(stage);
  stages.forEach((name, i) => {
    const el = $("step-" + name);
    el.className = (stage === "done" || i < currentIndex) ? "step done" : (i === currentIndex ? "step active" : "step");
    const metadata = $(name + "-meta");
    if (metadata && i === currentIndex && msg) metadata.textContent = msg;
  });
}
function stepProgress(name, percent, text) {
  const fill = $(name + "-fill"), label = $(name + "-pct"), meta = $(name + "-meta");
  if (fill) fill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  if (label) label.textContent = `${percent}%`;
  if (meta && text) meta.textContent = text;
}
function startUI() {
  clearTimeout(pollTimer);
  cleanupViewer();
  $("go").disabled = true;
  $("dl").hidden = true;
  $("downloadOptions").hidden = true;
  $("addonPanel").style.display = "none";
  $("qualityWarnings").hidden = true;
  $("stats").style.display = "none";
  $("viewerStats").hidden = true;
  $("viewer-ui").style.display = "none";
  $("viewer-info").style.display = "none";
  $("placeholder").style.display = "flex";
  $("placeholder").classList.add("processing");
  message("");
  stages.forEach(stage => { $("step-" + stage).className = "step"; });
  updateStep("queued", "Uploading video...");
  stepProgress("upload", 0);
  stepProgress("filter", 0);
  stepProgress("recon", 0);
}

const inputFile = $("file");
const dropZone = $("dropZone");
["dragenter", "dragover"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault(); dropZone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(name => dropZone.addEventListener(name, event => {
  event.preventDefault(); dropZone.classList.remove("dragover");
}));
dropZone.addEventListener("drop", event => {
  if (event.dataTransfer?.files?.length) {
    inputFile.files = event.dataTransfer.files;
    inspectVideo(event.dataTransfer.files[0]);
  }
});
inputFile.addEventListener("change", event => { if (event.target.files[0]) inspectVideo(event.target.files[0]); });
function inspectVideo(file) {
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|avi|webm)$/i.test(file.name)) {
    message("Please choose a video (MP4, MOV, AVI or WebM).", true); return;
  }
  if (cloudMaxUploadMB && file.size > cloudMaxUploadMB * 1048576) {
    message(`Video exceeds the cloud upload limit (${cloudMaxUploadMB} MB). Record a shorter H.264 MP4 clip.`, true);
    return;
  }
  if (originalVideoUrl) URL.revokeObjectURL(originalVideoUrl);
  originalVideoUrl = URL.createObjectURL(file);
  $("videoPreview").src = originalVideoUrl;
  $("compareVideo").src = originalVideoUrl;
  $("videoPreviewBox").style.display = "block";
  $("fileDetails").textContent = `${file.name} (${(file.size / 1048576).toFixed(1)} MB)`;
  $("videoAdvice").hidden = true;
  $("videoPreview").onloadedmetadata = () => {
    const v = $("videoPreview"), advice = [];
    if (Math.min(v.videoWidth, v.videoHeight) < 480) advice.push("Low resolution: 720p or higher is recommended.");
    if (v.duration < 6) advice.push("Short video: capture a longer, slower sweep.");
    if ($("mode").value === "turntable") advice.push("Turntable: keep the camera fixed, use a plain backdrop and rotate the object a full turn.");
    else advice.push("Orbit: keep the object still and move the camera around it.");
    $("videoAdvice").textContent = `Video ${v.videoWidth}×${v.videoHeight}, ${v.duration.toFixed(1)} sec.\n${advice.join("\n")}`;
    $("videoAdvice").hidden = false;
  };
}
$("mode").onchange = () => { if (inputFile.files[0]) inspectVideo(inputFile.files[0]); };

function cleanupViewer() {
  if (state.raf) cancelAnimationFrame(state.raf);
  if (state.controls && state.kind === "cloud") state.controls.dispose();
  if (state.renderer) {
    state.renderer.domElement.remove();
    state.renderer.dispose();
  }
  for (const item of [state.pointMesh, state.surfaceMesh]) item?.material?.dispose?.();
  state.originalGeometry?.dispose();
  state.surfaceGeometry?.dispose();
  if (state.splat) { state.splat.dispose(); }
  document.getElementById("gallery")?.remove();
  state = {kind: null, splat: null, renderer: null, scene: null, camera: null, controls: null,
    group: null, pointMesh: null, surfaceMesh: null, pointsMaterial: null, meshMaterial: null,
    originalGeometry: null, surfaceGeometry: null, raf: null, originalRotation: 0};
}
function showViewerUI(type) {
  $("placeholder").classList.remove("processing");
  $("placeholder").style.display = "none";
  $("viewer-ui").style.display = "flex";
  $("viewer-info").style.display = "block";
  $("model-type-badge").textContent = type;
}
function radialPointTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 32;
  const ctx = c.getContext("2d"), gradient = ctx.createRadialGradient(16,16,0,16,16,16);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.7, "rgba(255,255,255,1)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = gradient; ctx.fillRect(0,0,32,32);
  return new THREE.CanvasTexture(c);
}
let spriteTexture;

async function showCloud(job, status) {
  cleanupViewer();
  showViewerUI(status.model_type === "voxels" ? "Voxels + Surface" : "Point Cloud");
  const container = $("viewer"), width = Math.max(200, container.clientWidth), height = Math.max(200, container.clientHeight);
  const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height); renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  container.appendChild(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(darkCanvas ? 0x06090e : 0xf1f5f9);
  const camera = new THREE.PerspectiveCamera(50, width/height, 0.01, 2000);
  scene.add(new THREE.AmbientLight(0xffffff, 2.3));
  const key = new THREE.DirectionalLight(0xffffff, 2.7); key.position.set(2, 3, 5); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.7); fill.position.set(-3, 1, -3); scene.add(fill);
  const loader = new PLYLoader();
  const geometry = await loader.loadAsync(`/api/jobs/${job}/model.ply`);
  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere;
  if (!spriteTexture) spriteTexture = radialPointTexture();
  const pointsMaterial = new THREE.PointsMaterial({size: 3, sizeAttenuation: false,
    map: spriteTexture, alphaTest: 0.08, vertexColors: !!geometry.getAttribute("color"),
    transparent: true, opacity: 1, depthWrite: true});
  const pointMesh = new THREE.Points(geometry, pointsMaterial);
  pointMesh.position.copy(sphere.center).negate();
  const group = new THREE.Group();
  const originalRotation = status.model_type === "pointcloud" ? Math.PI : 0;
  group.rotation.x = originalRotation;
  group.add(pointMesh); scene.add(group);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.minDistance = Math.max(0.001, sphere.radius * 0.12);
  controls.maxDistance = Math.max(1, sphere.radius * 24);
  camera.position.set(0, 0, Math.max(sphere.radius * 2.6, .1));
  controls.saveState();
  state = {kind: "cloud", splat: null, renderer, scene, camera, controls, group,
    pointMesh, surfaceMesh: null, pointsMaterial, meshMaterial: null,
    originalGeometry: geometry, surfaceGeometry: null, raf: null, originalRotation};
  if (status.has_surface) {
    try {
      const meshGeo = await loader.loadAsync(`/api/jobs/${job}/model.ply?geometry=surface`);
      if (!meshGeo.getAttribute("normal")) meshGeo.computeVertexNormals();
      meshGeo.computeBoundingSphere();
      const meshMaterial = new THREE.MeshStandardMaterial({vertexColors: !!meshGeo.getAttribute("color"),
        side: THREE.DoubleSide, metalness: 0.0, roughness: 0.8, transparent: true});
      const surfaceMesh = new THREE.Mesh(meshGeo, meshMaterial);
      surfaceMesh.position.copy(meshGeo.boundingSphere.center).negate();
      group.add(surfaceMesh);
      state.surfaceMesh = surfaceMesh;
      state.meshMaterial = meshMaterial;
      state.surfaceGeometry = meshGeo;
    } catch (error) {
      message(`Surface preview was unavailable; points are still shown. ${error.message}`, true);
    }
  }
  const hasMesh = !!state.surfaceMesh;
  $("renderStyle").querySelectorAll("option").forEach(option => {
    option.disabled = (option.value !== "points" && !hasMesh);
  });
  $("renderStyle").value = hasMesh ? "mesh" : "points";
  $("meshNote").textContent = hasMesh ? "Solid mode is real triangulated geometry from silhouette carving." : "No triangle surface available in this reconstruction; points only.";
  $("addonPanel").style.display = "block";
  $("pointCountBadge").textContent = `${geometry.getAttribute("position").count.toLocaleString()} points`;
  const faces = hasMesh ? state.surfaceGeometry.index?.count / 3 : 0;
  $("viewerStats").textContent = `${geometry.getAttribute("position").count.toLocaleString()} points` + (faces ? ` • ${Math.round(faces).toLocaleString()} triangles` : " • point cloud");
  $("viewerStats").hidden = false;
  restorePreset();
  applyCustomizer();
  function loop() {
    if (state.kind !== "cloud") return;
    state.raf = requestAnimationFrame(loop);
    if (autoRotate) group.rotation.y += 0.006;
    controls.update(); renderer.render(scene, camera);
  }
  loop();
}

async function showGaussian(url) {
  cleanupViewer();
  showViewerUI("Gaussian Splat");
  $("addonPanel").style.display = "none";  // these materials aren't Three.js PointsMaterial
  $("viewerStats").textContent = "Gaussian splat • use orbit/zoom controls";
  $("viewerStats").hidden = false;
  const splat = new GaussianSplats3D.Viewer({
    rootElement: $("viewer"), cameraUp: [0,-1,0],
    initialCameraPosition: [0,-0.5,3], initialCameraLookAt: [0,0,0],
    sharedMemoryForWorkers: false
  });
  state.kind = "splat"; state.splat = splat;
  await splat.addSplatScene(url, {showLoadingUI: true, progressiveLoad: false});
  splat.start(); state.camera = splat.camera; state.controls = splat.controls;
  if (state.controls) { state.controls.autoRotate = autoRotate; state.controls.autoRotateSpeed = 0.8; }
}

function applyCustomizer() {
  if (state.kind !== "cloud") return;
  const val = id => Number($(id).value);
  const style = $("renderStyle").value;
  const displayMesh = style !== "points" && !!state.surfaceMesh;
  state.pointMesh.visible = !displayMesh;
  if (state.surfaceMesh) state.surfaceMesh.visible = displayMesh;
  let size = val("pointSizeRange");
  if ($("adaptiveClarity").checked) size = Math.min(size, 3.5);
  state.pointsMaterial.size = size;
  state.pointsMaterial.opacity = val("opacityRange");
  state.pointsMaterial.depthWrite = val("opacityRange") === 1;
  state.pointsMaterial.color.setScalar(val("brightnessRange"));
  state.pointsMaterial.needsUpdate = true;
  if (state.meshMaterial) {
    state.meshMaterial.opacity = val("opacityRange");
    state.meshMaterial.depthWrite = val("opacityRange") === 1;
    state.meshMaterial.color.setScalar(val("brightnessRange"));
    state.meshMaterial.roughness = Math.max(.15, 1 - val("finishRange") / 125);
    state.meshMaterial.wireframe = style === "wireframe";
    state.meshMaterial.needsUpdate = true;
  }
  state.group.rotation.set(
    state.originalRotation + THREE.MathUtils.degToRad(val("rotateX")),
    THREE.MathUtils.degToRad(val("rotateY")),
    THREE.MathUtils.degToRad(val("rotateZ"))
  );
  state.group.scale.setScalar(val("scaleRange"));
  $("pointSizeValue").textContent = `${val("pointSizeRange")}px`;
  $("opacityValue").textContent = `${Math.round(100 * val("opacityRange"))}%`;
  $("brightnessValue").textContent = `${Math.round(100 * val("brightnessRange"))}%`;
  $("finishValue").textContent = `${val("finishRange")}%`;
  for (const axis of ["X","Y","Z"]) $("rotate"+axis+"Value").textContent = `${val("rotate"+axis)}°`;
  $("scaleValue").textContent = `${val("scaleRange").toFixed(2)}×`;
}
for (const id of settingIds) $(id).addEventListener(id === "renderStyle" || id === "adaptiveClarity" ? "change" : "input", applyCustomizer);
$("resetAppearance").onclick = () => {
  for (const [id, value] of Object.entries(defaultSettings)) {
    if (id === "adaptiveClarity") $(id).checked = value;
    else $(id).value = value;
  }
  if (state.surfaceMesh) $("renderStyle").value = "mesh";
  applyCustomizer();
};
$("savePreset").onclick = () => {
  const values = Object.fromEntries(settingIds.map(id => [id, id === "adaptiveClarity" ? $(id).checked : $(id).value]));
  localStorage.setItem("srujana-customizer-preset", JSON.stringify(values));
  message("Customizer preset saved in this browser.");
};
function restorePreset() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem("srujana-customizer-preset") || "null"); }
  catch (_) { saved = null; }
  if (!saved) return;
  for (const id of settingIds) {
    if (!Object.prototype.hasOwnProperty.call(saved, id)) continue;
    if (id === "adaptiveClarity") $(id).checked = !!saved[id];
    else if (id === "renderStyle") $(id).value = saved[id] !== "points" && !state.surfaceMesh ? "points" : saved[id];
    else $(id).value = saved[id];
  }
}

async function showGallery(id) {
  cleanupViewer();
  showViewerUI("Sharp Frames Only");
  $("viewer-ui").style.display = "none";
  const resp = await fetch(`/api/jobs/${id}/frames`);
  if (!resp.ok) throw apiError(resp, await resp.json().catch(() => null));
  const data = await resp.json();
  const gallery = document.createElement("div"); gallery.id = "gallery";
  gallery.style.cssText = "position:absolute;inset:0;overflow:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;padding:1.5rem;align-content:start;background:#fff;z-index:5";
  data.frames.forEach(file => {
    const img = document.createElement("img"); img.loading = "lazy";
    img.src = `/api/jobs/${id}/frames/${encodeURIComponent(file)}`;
    img.style.cssText = "width:100%;border-radius:8px";
    gallery.appendChild(img);
  });
  $("viewer").appendChild(gallery);
}

function setupDownloads(id, hasSurface, modelType) {
  $("dl").hidden = false;
  $("downloadMenu").hidden = false;
  $("dlFrames").hidden = true;
  $("dlPly").href = `/api/jobs/${id}/model.ply`;
  for (const ext of ["MeshPly", "Glb", "Stl"]) $("dl" + ext).hidden = !hasSurface;
  if (hasSurface) {
    $("dlMeshPly").href = `/api/jobs/${id}/model.ply?geometry=surface`;
    $("dlGlb").href = `/api/jobs/${id}/model.glb?geometry=surface`;
    $("dlStl").href = `/api/jobs/${id}/model.stl?geometry=surface`;
  }
  $("downloadGeometry").querySelector('option[value="surface"]').disabled = !hasSurface;
  $("downloadGeometry").value = hasSurface ? "surface" : "original";
  $("downloadGeometry").onchange = () => updateMeshDownloadLinks(id);
  updateMeshDownloadLinks(id);
}
function updateMeshDownloadLinks(id) {
  const shape = $("downloadGeometry").value;
  for (const ext of ["obj", "blend"]) {
    const el = $("dl" + ext[0].toUpperCase() + ext.slice(1));
    el.href = `/api/jobs/${id}/model.${ext}?geometry=${shape}`;
    el.download = (shape === "surface" ? "surface" : "model") + "." + ext;
  }
}
$("downloadToggle").onclick = () => {
  $("downloadOptions").hidden = !$("downloadOptions").hidden;
  $("downloadToggle").setAttribute("aria-expanded", String(!$("downloadOptions").hidden));
};
document.addEventListener("click", event => {
  if (!$("downloadMenu").contains(event.target)) $("downloadOptions").hidden = true;
});
for (const id of ["dlPly","dlMeshPly","dlObj","dlBlend","dlGlb","dlStl"]) {
  $(id).addEventListener("click", () => { $("downloadOptions").hidden = true; });
}

function renderJobInfo(status) {
  const stats = status.stats || {};
  const parts = [];
  if (stats.kept_frames != null) parts.push(`Frames: ${stats.kept_frames} selected / ${stats.total_frames}`);
  if (stats.points != null) parts.push(`Points: ${stats.points.toLocaleString()}`);
  if (stats.mesh?.faces) parts.push(`Faces: ${stats.mesh.faces.toLocaleString()}`);
  if (stats.grid_resolution) parts.push(`Grid: ${stats.grid_resolution}`);
  if (parts.length) { $("stats").style.display = "block"; $("stats").textContent = parts.join("  •  "); }
  const report = status.quality_report;
  const lines = report ? [
    `Video diagnostics: ${report.width}×${report.height} • ${report.duration_seconds}s • sampled sharpness ${report.median_sharpness}`,
    ...(report.warnings || [])
  ] : [];
  if (stats.surface_error) lines.push(stats.surface_error);
  $("qualityWarnings").textContent = lines.join("\n");
  $("qualityWarnings").hidden = !lines.length;
}
async function displayFinished(id, status) {
  currentJob = id;
  // History reopens the exact original video for comparison, not an unrelated
  // video that happens to be selected in the upload form.
  $("compareVideo").src = `/api/jobs/${id}/video`;
  $("compareVideo").hidden = true;
  $("go").disabled = false;
  $("placeholder").classList.remove("processing");
  updateStep("done", status.message);
  stepProgress("filter", 100, "Sharp frames extracted");
  stepProgress("recon", 100, "Finished");
  renderJobInfo(status);
  message(status.message || "Reconstruction ready.");
  if (status.has_model) {
    setupDownloads(id, !!status.has_surface, status.model_type);
    if (status.model_type === "splat") await showGaussian(`/api/jobs/${id}/model.ply`);
    else await showCloud(id, status);
  } else {
    $("dl").hidden = false;
    $("downloadMenu").hidden = true;
    $("dlFrames").hidden = false;
    $("dlFrames").href = `/api/jobs/${id}/frames.zip`;
    await showGallery(id);
  }
  loadRecentJobs();
}
async function poll(id) {
  clearTimeout(pollTimer);
  try {
    const response = await fetch(`/api/jobs/${id}`);
    if (!response.ok) throw apiError(response, await response.json().catch(() => null));
    const status = await response.json();
    updateStep(status.stage, status.message);
    if (status.stage === "failed") {
      $("go").disabled = false; $("placeholder").classList.remove("processing");
      message(status.message, true); return;
    }
    if (status.stage === "done") { await displayFinished(id, status); return; }
    if (status.stage === "filtering") stepProgress("filter", 45, status.message);
    if (status.stage === "reconstructing") {
      stepProgress("filter", 100, "Sharp frames selected");
      stepProgress("recon", status.message.includes("3/3") ? 75 : status.message.includes("Carving") ? 55 : 30, status.message);
    }
    pollTimer = setTimeout(() => poll(id), 1800);
  } catch (error) {
    $("go").disabled = false;
    $("placeholder").classList.remove("processing");
    message(error.message, true);
  }
}
$("go").addEventListener("click", async () => {
  const file = inputFile.files[0];
  if (!file) { message("Choose a video first.", true); return; }
  if (cloudMaxUploadMB && file.size > cloudMaxUploadMB * 1048576) {
    message(`Video exceeds ${cloudMaxUploadMB} MB. Compress or trim it before uploading.`, true); return;
  }
  startUI();
  const form = new FormData();
  form.append("video", file); form.append("mode", $("mode").value);
  form.append("quality", $("qualityPreset").value);
  form.append("remove_noise", String($("removeNoise").checked));
  try {
    const jobId = await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open("POST", "/api/jobs");
      request.upload.onprogress = event => {
        if (event.lengthComputable) stepProgress("upload", Math.round(100*event.loaded/event.total), "Uploading video...");
      };
      request.onload = () => {
        let result;
        try { result = JSON.parse(request.responseText); }
        catch (_) { result = {}; }
        if (request.status >= 200 && request.status < 300) resolve(result.id);
        else reject(new Error(result.detail || `Upload failed (${request.status}); check the backend terminal.`));
      };
      request.onerror = () => reject(new Error("Cannot reach the 3D service. Check your internet connection or refresh the app."));
      request.send(form);
    });
    stepProgress("upload", 100, "Upload complete");
    await poll(jobId);
  } catch (error) {
    message(error.message, true);
    $("go").disabled = false; $("placeholder").classList.remove("processing");
  }
});
async function loadRecentJobs() {
  try {
    const resp = await fetch("/api/jobs?limit=15");
    if (!resp.ok) return;
    const {jobs} = await resp.json();
    const selector = $("recentJobs"); selector.replaceChildren();
    if (!jobs.length) selector.add(new Option("No previous jobs", ""));
    for (const job of jobs) {
      const date = new Date((job.updated || 0)*1000).toLocaleString();
      selector.add(new Option(`${date} — ${job.stage} — ${job.id}`, job.id));
    }
  } catch (_) { /* history is optional when backend is offline */ }
}
$("openJob").onclick = async () => {
  const id = $("recentJobs").value;
  if (!id) return;
  clearTimeout(pollTimer); cleanupViewer();
  try { await poll(id); }
  catch (error) { message(error.message, true); }
};
loadRecentJobs();

function zoom(factor) {
  if (!state.camera) return;
  const target = state.controls?.target || new THREE.Vector3();
  state.camera.position.sub(target).multiplyScalar(factor).add(target);
  state.camera.updateProjectionMatrix();
}
$("zoomIn").onclick = () => zoom(.8);
$("zoomOut").onclick = () => zoom(1.25);
$("resetCam").onclick = () => {
  if (state.kind === "splat") {
    state.camera?.position.set(0,-.5,3);
    state.controls?.target.set(0,0,0);
  } else if (state.controls) {
    state.controls.reset();
    for (const id of ["rotateX","rotateY","rotateZ"]) $(id).value = "0";
    $("scaleRange").value = "1"; applyCustomizer();
  }
};
$("autoRotate").onclick = () => {
  autoRotate = !autoRotate;
  if (state.kind === "splat" && state.controls) {
    state.controls.autoRotate = autoRotate;
    state.controls.autoRotateSpeed = 0.8;
  }
  $("autoRotate").classList.toggle("selected", autoRotate);
};
$("toggleBg").onclick = () => {
  darkCanvas = !darkCanvas;
  $("viewer").style.background = darkCanvas ? "#06090e" : "#f1f5f9";
  if (state.scene) state.scene.background = new THREE.Color(darkCanvas ? 0x06090e : 0xf1f5f9);
};
$("compareBtn").onclick = () => {
  if (!currentJob && !originalVideoUrl) { message("Upload a source video to compare it with the model.", true); return; }
  const video = $("compareVideo"); video.hidden = !video.hidden;
  if (video.hidden) video.pause();
};
$("fullScreen").onclick = async () => {
  try { if (!document.fullscreenElement) await $("viewer").requestFullscreen();
    else await document.exitFullscreen(); }
  catch (error) { message(error.message, true); }
};
$("snapshotBtn").onclick = () => {
  const canvas = state.renderer?.domElement || state.splat?.renderer?.domElement;
  if (!canvas) { message("Snapshot isn't available for the current preview.", true); return; }
  try {
    const anchor = document.createElement("a");
    anchor.download = `srujana-${currentJob || "model"}.png`;
    anchor.href = canvas.toDataURL("image/png"); anchor.click();
  } catch (error) { message("Could not capture snapshot: " + error.message, true); }
};
document.addEventListener("keydown", event => {
  if (event.target.matches("input,select,textarea") || !state.camera) return;
  const shortcut = {"+":"zoomIn", "=":"zoomIn", "-":"zoomOut", "r":"resetCam", "a":"autoRotate"}[event.key.toLowerCase()];
  if (shortcut) $(shortcut).click();
});
function resizeViewer() {
  if (!state.renderer || !state.camera) return;
  const width = Math.max(200,$("viewer").clientWidth), height = Math.max(200,$("viewer").clientHeight);
  state.camera.aspect = width/height; state.camera.updateProjectionMatrix();
  state.renderer.setSize(width, height);
}
new ResizeObserver(resizeViewer).observe($("viewer"));
