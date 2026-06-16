import { BrowserRouter, Routes, Route } from "react-router-dom";
import HeaderBar from "./components/HeaderBar";
import Home from "./pages/Home";
import ExpeditionsDashboard from "./pages/ExpeditionsDashboard";
import ExpeditionDetail from "./pages/ExpeditionDetail";
import CFGPage from "./pages/CFGPage";

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <HeaderBar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/expeditions" element={<ExpeditionsDashboard />} />
          <Route path="/expeditions/:id" element={<ExpeditionDetail />} />
          <Route path="/CFG" element={<CFGPage />} />
          <Route path="/CFG/:id" element={<CFGPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;