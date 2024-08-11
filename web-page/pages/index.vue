<script setup lang="ts">
useSeoMeta({
  title: "Typress Demo",
});

// select backend
enum BackendType {
  Ndarray,
  Candle,
  Webgpu,
}

const backendSelect = ref(BackendType.Ndarray);

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
let isMainLoading = ref(true);
let isResLoading = ref(false);
let mainLoadingInfo = ref("正在加載資源...");
// execution time
let start = null;
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
  if (backendSelect.value === BackendType.Ndarray) {
    reset(true);
    worker?.postMessage({ type: "set_backend_ndarray", data: "" });
  } else if (backendSelect.value === BackendType.Candle) {
    reset(true);
    worker?.postMessage({ type: "set_backend_candle", data: "" });
  } else {
    reset(true);
    worker?.postMessage({ type: "set_backend_wgpu", data: "" });
  }
});

async function handleImageChange(event: Event) {
  let target = event.target as HTMLInputElement;
  reset(false);
  isResLoading.value = true;

  if (target.files && target.files.length > 0) {
    let file = target.files[0];

    let reader = new FileReader();
    reader.onload = async (event) => {
      if (imageDisplay.value !== null) {
        let img = imageDisplay.value;
        img.src = event.target?.result as string;
        await new Promise((resolve) => {
          img.onload = resolve;
        });

        // make sure the image is displayed on the page immediately after uploading it
        ctx?.clearRect(0, 0, 384, 384);
        // display the image on the canvas and resize the image
        ctx?.drawImage(img, 0, 0, 384, 384);
        if (imageCanvas.value !== null && ctx !== null) {
          let imageData = extractRGBValuesFromCanvas(imageCanvas.value, ctx);

          if (worker) {
            worker.onmessage = function (event) {
              if (event.data.status == "success") {
                resString.value = event.data.data;
                isResLoading.value = false;
                timeCost.value = performance.now() - start;
                console.log(resString.value);
              }
            };
            let start = performance.now();
            worker.postMessage({ type: "inference", data: imageData });
          }
        }
      }
    };

    reader.readAsDataURL(file);
  }
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
      <div class="h-screen pt-32">
        <div class="mb-2 flex items-center gap-4">
          <!-- Backend Selection -->
          <div>
            <label for="backend">後端：</label>
            <select
              id="backend"
              class="rounded-md border-[1.5px] border-gray-300"
              :disabled="isResLoading"
              v-model="backendSelect"
            >
              <option :value="BackendType.Ndarray" selected>
                CPU - Ndarray
              </option>
              <option :value="BackendType.Candle">CPU - Candle</option>
              <option :value="BackendType.Webgpu">GPU - WebGPU</option>
            </select>
          </div>

          <!-- Image Input -->
          <div>
            <input
              type="file"
              @change="handleImageChange"
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
          />
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
