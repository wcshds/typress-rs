let trocr = null;

async function init() {
  const { default: wasm, TrOCR } = await import(
    "../packages/typress-pkg/simd/typress_web"
  );
  await wasm();
  try {
    trocr = new TrOCR();
    self.postMessage({ status: "success", data: "" });
  } catch {
    self.postMessage({ status: "fail", data: "" });
  }
}

async function set_backend_ndarray() {
  try {
    await trocr?.set_backend_ndarray();
    self.postMessage({ status: "success", data: "" });
  } catch {
    self.postMessage({ status: "fail", data: "" });
  }
}

async function set_backend_candle() {
  try {
    await trocr?.set_backend_candle();
    self.postMessage({ status: "success", data: "" });
  } catch {
    self.postMessage({ status: "fail", data: "" });
  }
}

async function set_backend_wgpu() {
  try {
    await trocr?.set_backend_wgpu();
    self.postMessage({ status: "success", data: "" });
  } catch {
    self.postMessage({ status: "fail", data: "" });
  }
}

async function inference(imageData) {
  try {
    let res = await trocr?.inference(imageData);
    self.postMessage({ status: "success", data: res });
  } catch {
    self.postMessage({ status: "fail", data: "" });
  }
}

self.onmessage = async function (event) {
  const { type, data } = event.data;
  switch (type) {
    case "init":
      init();
      break;
    case "set_backend_ndarray":
      await set_backend_ndarray();
      break;
    case "set_backend_candle":
      await set_backend_candle();
      break;
    case "set_backend_wgpu":
      await set_backend_wgpu();
      break;
    case "inference":
      await inference(data);
      break;
    default:
      console.error("Unknown task type:", type);
  }
};
