<script setup lang="ts">
import { normalizeColor, type RGBList } from "./_internal/processColor";

const props = defineProps<{
  placeholder?: string;
  color?: RGBList | string;
  modelValue?: string;
}>();

// 發送按鈕事件到父組件
defineEmits<{
  search: [];
  "update:modelValue": [payload: string];
}>();

let normalised_color_str_100_opacity = ref("");
let normalised_color_str_80_opacity = ref("");
let normalised_color_str_70_opacity = ref("");
let normalised_color_str_60_opacity = ref("");
let normalised_color_str_20_opacity = ref("");

watch(
  () => props.color,
  () => {
    let {
      color_str_100_opacity,
      color_str_80_opacity,
      color_str_70_opacity,
      color_str_60_opacity,
      color_str_20_opacity,
    } = normalizeColor(props.color, [8, 145, 178]);

    normalised_color_str_100_opacity.value = color_str_100_opacity;
    normalised_color_str_80_opacity.value = color_str_80_opacity;
    normalised_color_str_70_opacity.value = color_str_70_opacity;
    normalised_color_str_60_opacity.value = color_str_60_opacity;
    normalised_color_str_20_opacity.value = color_str_20_opacity;
  },
  { immediate: true },
);
</script>

<template>
  <textarea
    class="input-color h-full w-full rounded-md border-[1.5px] px-2 py-1.5 outline-none transition duration-300 ease-in-out focus:ring-2"
    :value="modelValue"
    @input="
      $emit('update:modelValue', ($event.target as HTMLInputElement).value)
    "
  ></textarea>
</template>

<style scoped>
.input-color {
  caret-color: v-bind(normalised_color_str_80_opacity);
}
.input-color:hover {
  border-color: v-bind(normalised_color_str_80_opacity);
}
.input-color:focus {
  border-color: v-bind(normalised_color_str_60_opacity);
  --tw-ring-color: v-bind(normalised_color_str_20_opacity);
}
</style>
