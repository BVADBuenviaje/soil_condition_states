module.exports = {
  content: ["../app/index.html", "../app/app.js"],
  theme: {
    extend: {
      colors: {
        canvas: "#F3F5F1",
        card: "#FFFFFF",
        raised: "#F7F9F6",
        line: "#DCE2DA",
        ink: "#14201A",
        muted: "#4B5852",
        dim: "#67736C",
        brand: "#276B47",
        clay: "#A4512B",
        s1: "#2F62A0",
        s2: "#2C7D53",
        s3: "#9A6A0C",
        s4: "#AE4724",
        empty: "#E6EAE4",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "Segoe UI", "Arial", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "Consolas", "monospace"],
      },
      keyframes: {
        viewIn: { "0%": { opacity: "0", transform: "translateY(8px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
        fadeIn: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        breathe: { "0%, 100%": { transform: "scale(1)", opacity: "1" }, "50%": { transform: "scale(1.04)", opacity: "0.85" } },
        slide: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(250%)" } },
      },
      animation: {
        viewIn: "viewIn 260ms cubic-bezier(.2,.7,.2,1) both",
        fadeIn: "fadeIn 220ms ease-out both",
        breathe: "breathe 1.8s ease-in-out infinite",
        slide: "slide 1.1s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
