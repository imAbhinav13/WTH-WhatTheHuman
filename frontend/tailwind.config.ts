import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "bg-base": "#EFE3D1",
        "bg-raised": "#FAF5EA",
        ink: "#4A2E1E",
        accent: "#A6522F",
        danger: "#7A2E24",
      },
      fontFamily: {
        serif: ["var(--font-eb-garamond)", "Georgia", "serif"],
        sans: ["var(--font-ui)", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      maxWidth: {
        reading: "48rem",
      },
      lineHeight: {
        reading: "1.68",
      },
    },
  },
  plugins: [],
};

export default config;
