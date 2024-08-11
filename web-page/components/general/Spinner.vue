<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    active: boolean;
    info?: string;
  }>(),
  { info: "" },
);
</script>

<template>
  <div class="relative">
    <div
      class="absolute left-1/2 top-[45%] z-50 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-2"
      v-show="active"
    >
      <svg
        class="z-30 w-8 animate-spin text-cyan-700/90"
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        <circle
          fill="none"
          stroke="currentColor"
          stroke-width="20"
          stroke-linecap="round"
          cx="100"
          cy="100"
          r="90"
          stroke-dasharray="491"
          stroke-dashoffset="246"
        ></circle>
      </svg>
      <span class="text-cyan-600">{{ info }}</span>
    </div>

    <div :class="[active ? 'opacity-50' : '']">
      <slot></slot>
    </div>
  </div>
</template>
