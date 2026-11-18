const express = require("express");
const path = require("path");
const { createProxyMiddleware } = require("http-proxy-middleware");
const app = express();
const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";
// Express removes the /api mount prefix before the proxy handles a request.
// FastAPI routes include /api (for example POST /api/jobs), so restore it.
app.use("/api", createProxyMiddleware({
  target: backend,
  changeOrigin: true,
  pathRewrite: (pathname) => "/api" + pathname,
}));
app.use(express.static(path.join(__dirname, "public")));
app.listen(process.env.PORT || 3000, () => console.log("Srujana frontend http://localhost:" + (process.env.PORT || 3000)));
