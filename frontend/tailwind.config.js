/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'medical-blue': '#0066cc',
        'medical-dark': '#1a1a2e',
        'medical-gray': '#2d3748',
        'tumor-necrotic': '#ef4444',
        'tumor-edema': '#22c55e',
        'tumor-enhancing': '#eab308',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
      }
    },
  },
  plugins: [],
}
