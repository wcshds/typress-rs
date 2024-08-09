<script setup lang="ts">
import { onMounted, ref, watch, type Ref } from "vue";
import type { TrOCR } from "../typress-pkg/simd/typress_web";

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
// typress inference model
let trocr: TrOCR | null = null;
// typress result string
let resString = ref("");

onMounted(async () => {
  ctx = imageCanvas.value?.getContext("2d", {
    willReadFrequently: true,
  }) as CanvasRenderingContext2D;

  // wasm dynamic import
  const { default: wasm, TrOCR } = await import(
    "../typress-pkg/simd/typress_web"
  );

  wasm().then(() => {
    trocr = new TrOCR();
  });
});

watch(backendSelect, async () => {
  if (backendSelect.value === BackendType.Ndarray) {
    reset();
    await trocr?.set_backend_ndarray();
  } else if (backendSelect.value === BackendType.Candle) {
    reset();
    await trocr?.set_backend_candle();
  } else {
    reset();
    await trocr?.set_backend_wgpu();
  }
});

async function handleImageChange(event: Event) {
  let target = event.target as HTMLInputElement;
  resString.value = "";
  if (imageDisplay.value !== null) {
    imageDisplay.value.src = "";
  }

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
        setTimeout(async () => {
          ctx?.clearRect(0, 0, 384, 384);
          // display the image on the canvas and resize the image
          ctx?.drawImage(img, 0, 0, 384, 384);
          if (imageCanvas.value !== null && ctx !== null) {
            let imageData = extractRGBValuesFromCanvas(imageCanvas.value, ctx);
            // nullish coalescing operator
            resString.value = (await trocr?.inference(imageData)) ?? "";
            console.log(resString.value);
          }
        }, 20);
      }
    };

    reader.readAsDataURL(file);
  }
}

function reset() {
  resString.value = "";
  if (imageDisplay.value !== null) {
    imageDisplay.value.src = "";
  }
  if (imageInput.value !== null) {
    imageInput.value.value = "";
  }
}

function extractRGBValuesFromCanvas(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D
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

  <main class="px-5 py-20">
    <!-- Backend Selection -->
    <label for="backend">後端：</label>
    <select
      id="backend"
      class="rounded-md border-2 border-gray-300"
      v-model="backendSelect"
    >
      <option :value="BackendType.Ndarray" selected>CPU - Ndarray</option>
      <option :value="BackendType.Candle">CPU - Candle</option>
      <option :value="BackendType.Webgpu">GPU - WebGPU</option>
    </select>

    <!-- Image Input -->
    <div>
      <input
        type="file"
        @change="handleImageChange"
        accept="image/*"
        ref="imageInput"
      />
    </div>

    <!-- the actual <img> used to display image -->
    <div
      class="flex h-[200px] w-full items-center justify-center rounded-md border-2"
    >
      <img
        ref="imageDisplay"
        class="max-h-[200px] max-w-[384px] rounded-md border-2"
      />
    </div>

    <div>
      識別結果：<br />
      <div class="px-14">
        <textarea
          class="h-[150px] w-full rounded-md border-2 p-2"
          v-model="resString"
        ></textarea>
      </div>
    </div>

    <!-- a hidden canvas used to resize image -->
    <canvas ref="imageCanvas" class="hidden" width="384" height="384"></canvas>
  </main>
</template>

<style scoped></style>
