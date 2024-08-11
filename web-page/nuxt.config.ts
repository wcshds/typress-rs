// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: "2024-04-03",
  devtools: { enabled: true },
  modules: ["@nuxtjs/tailwindcss"],
  components: [
    {
      path: "~/components",
      extensions: [".vue"],
    },
    {
      path: "~/components/general/",
      prefix: "g",
    },
  ],
  vite: {
    worker: {
      format: "es",
    },
  },
});
