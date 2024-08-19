<script setup lang="ts">
useSeoMeta({
  title: "Typress Demo",
});

// select backend
enum BackendType {
  Candle,
  Ndarray,
  Webgpu,
}

const backendSelect = ref(BackendType.Candle);

// ref dom
const imageCanvas: Ref<HTMLCanvasElement | null> = ref(null);
const imageDisplay: Ref<HTMLImageElement | null> = ref(null);
const imageInput: Ref<HTMLInputElement | null> = ref(null);
// canvas context
let ctx: CanvasRenderingContext2D | null = null;
// web worker
let worker: Worker | null = null;
// typress result string
let resString = ref("");
// flags
let useRustResizeImage = ref(true); // Whether to use rust to resize the image
let isMainLoading = ref(true);
let isResLoading = ref(false);
let mainLoadingInfo = ref("正在加載資源...");
// execution time
let start = 0;
let timeCost = ref(Infinity);

onMounted(async () => {
  isMainLoading.value = true;
  mainLoadingInfo.value = "正在加載資源...";
  ctx = imageCanvas.value?.getContext("2d", {
    willReadFrequently: true,
  }) as CanvasRenderingContext2D;

  worker = new Worker(new URL("../worker/trocr.js", import.meta.url));
  // 处理 WebAssembly 计算
  worker.onmessage = function (event) {
    if (event.data.status == "success") {
      isMainLoading.value = false;
    }
  };
  worker.postMessage({ type: "init", data: "" });

  document.addEventListener("paste", handleImagePaste);
});

watch(backendSelect, () => {
  mainLoadingInfo.value = "正在加載模型...";
  isMainLoading.value = true;
  if (worker) {
    worker.onmessage = function (event) {
      if (event.data.status == "success") {
        isMainLoading.value = false;
      }
    };
  }
  if (backendSelect.value === BackendType.Candle) {
    reset(true);
    worker?.postMessage({ type: "set_backend_candle", data: "" });
  } else if (backendSelect.value === BackendType.Ndarray) {
    reset(true);
    worker?.postMessage({ type: "set_backend_ndarray", data: "" });
  } else {
    reset(true);
    worker?.postMessage({ type: "set_backend_wgpu", data: "" });
  }
});

async function handleImageFileUpload(event: Event) {
  let target = event.target as HTMLInputElement;

  if (!target.files) return;
  if (target.files.length <= 0) return;

  reset(false);
  isResLoading.value = true;

  let file = target.files[0];
  doInference(file);
}

async function handleImagePaste(event: ClipboardEvent) {
  if (isMainLoading.value || isResLoading.value) return;

  if (!event.clipboardData) return;

  const pText = event.clipboardData.getData("text/plain");
  if (pText) return;

  const items = event.clipboardData.items;
  if (
    !items ||
    items[0].kind !== "file" ||
    items[0].type.indexOf("image") === -1
  ) {
    console.log("No image file in clipboard data.");
    return;
  }

  const blob = items[0].getAsFile();
  if (!blob) {
    console.log("Failed to convert pasted file.");
    return;
  }

  reset(true);
  isResLoading.value = true;
  doInference(blob);
}

function doInference(file: Blob) {
  let reader = new FileReader();
  reader.onload = async (event) => {
    if (imageDisplay.value !== null) {
      let img = imageDisplay.value;
      img.src = event.target?.result as string;
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      if (!worker) return;

      let imageData;
      if (useRustResizeImage.value) {
        imageData = base64ToUint8Array(img.src);
      } else {
        ctx?.clearRect(0, 0, 384, 384);
        // display the image on the canvas and resize the image
        ctx?.drawImage(img, 0, 0, 384, 384);
        if (imageCanvas.value !== null && ctx !== null) {
          imageData = extractRGBValuesFromCanvas(imageCanvas.value, ctx);
        }
      }

      worker.onmessage = function (event) {
        if (event.data.status == "success") {
          resString.value = event.data.data;
          isResLoading.value = false;
          timeCost.value = performance.now() - start;
          console.log(resString.value);
        }
      };
      start = performance.now();
      worker.postMessage({
        type: "inference",
        data: {
          imageData: imageData,
          isRawImageData: useRustResizeImage.value,
        },
      });
    }
  };

  reader.readAsDataURL(file);
}

function reset(resetImageInput: Boolean = false) {
  resString.value = "";
  timeCost.value = Infinity;
  if (imageDisplay.value !== null) {
    imageDisplay.value.src = "";
  }
  if (resetImageInput && imageInput.value !== null) {
    imageInput.value.value = "";
  }
}

function base64ToUint8Array(base64: string): Uint8Array {
  const base64String = base64.split(",")[1];

  const binaryString = window.atob(base64String);
  const length = binaryString.length;
  const bytes = new Uint8Array(length);

  for (let i = 0; i < length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  return bytes;
}

function extractRGBValuesFromCanvas(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
) {
  // Get image data from the canvas
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // Get canvas dimensions
  const height = canvas.height;
  const width = canvas.width;

  // Create a flattened array to hold the RGB values in channel-first order
  const flattenedArray = new Float32Array(3 * height * width);

  // Initialize indices for R, G, B channels in the flattened array
  let indexRed = 0,
    indexGreen = height * width,
    indexBlue = 2 * height * width;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Compute the index for the image data array
      const index = (y * width + x) * 4;

      // Fill in the R, G, B channels in the flattened array
      flattenedArray[indexRed++] = imageData.data[index] / 255.0; // Red
      flattenedArray[indexGreen++] = imageData.data[index + 1] / 255.0; // Green
      flattenedArray[indexBlue++] = imageData.data[index + 2] / 255.0; // Blue
    }
  }

  return flattenedArray;
}
</script>

<template>
  <header></header>

  <main class="container mx-auto">
    <GSpinner :active="isMainLoading" :info="mainLoadingInfo">
      <div class="h-screen pt-28">
        <div class="mb-2 flex flex-col items-start gap-4">
          <!-- Backend Selection -->
          <div>
            <label for="backend">後端：</label>
            <select
              id="backend"
              class="rounded-md border-[1.5px] border-gray-300"
              :disabled="isResLoading"
              v-model="backendSelect"
            >
              <option :value="BackendType.Candle" selected>CPU - Candle</option>
              <option :value="BackendType.Ndarray">CPU - Ndarray</option>
              <option :value="BackendType.Webgpu">GPU - WebGPU</option>
            </select>
          </div>

          <!-- Whether to use rust to resize the image -->
          <div
            class="flex cursor-pointer select-none gap-2"
            :disabled="isResLoading"
          >
            <input
              type="checkbox"
              id="checkbox"
              class="cursor-pointer"
              v-model="useRustResizeImage"
              :disabled="isResLoading"
            />
            <label
              for="checkbox"
              class="cursor-pointer disabled:opacity-50"
              :disabled="isResLoading"
            >
              調整圖像大小時是否使用Rust
            </label>
          </div>

          <!-- Image Input -->
          <div>
            <input
              type="file"
              @change="handleImageFileUpload"
              accept="image/*"
              :disabled="isResLoading"
              ref="imageInput"
            />
          </div>
        </div>

        <!-- the actual <img> used to display image -->
        <div
          class="mb-2 flex h-[200px] w-full items-center justify-center rounded-md border-[1.5px]"
        >
          <img
            ref="imageDisplay"
            class="max-h-[200px] max-w-[384px] rounded-md border-[1.5px]"
            v-show="isResLoading || resString.length > 0"
          />
          <div
            class="flex items-center gap-2"
            v-show="(!isResLoading && resString.length === 0) || isMainLoading"
          >
            <IconInformation
              class="w-8 flex-shrink-0 text-sky-500"
            ></IconInformation>
            <p class="select-none text-justify text-xl text-slate-600">
              上傳圖片進行公式識别，或
              <span class="text-nowrap">Ctrl + V</span>
              可識別剪切板內的圖片
            </p>
          </div>
        </div>

        <div>
          識別結果：<br />
          <GSpinner :active="isResLoading" info="正在識別公式...">
            <div class="px-10">
              <GTextbox v-model="resString" class="min-h-[150px]"></GTextbox>
              <span :class="isFinite(timeCost) ? '' : 'hidden'">
                識別共耗時：{{ (timeCost / 1000).toPrecision(3) }} 秒
              </span>
            </div>
          </GSpinner>
        </div>

        <!-- a hidden canvas used to resize image -->
        <canvas
          ref="imageCanvas"
          class="hidden"
          width="384"
          height="384"
        ></canvas>
      </div>
    </GSpinner>
  </main>
</template>

<style scoped></style>
