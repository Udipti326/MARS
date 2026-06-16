import { NavLink, Link } from "react-router-dom";

function HeaderBar() {
  const base = ({ isActive }) => `nav-link ${isActive ? "nav-link-active" : ""}`;

  return (
    <header className="topbar">
      <div>
        <Link to="/" className="brand">
          MARS
        </Link>
        <div className="brand-subtitle">Multi-Agent Research System</div>
      </div>

      <nav className="nav">
        <NavLink to="/" className={base}>
          Research
        </NavLink>
        <NavLink to="/expeditions" className={base}>
          Expeditions
        </NavLink>
        <NavLink to="/CFG" className={base}>
          CFG
        </NavLink>
      </nav>
    </header>
  );
}

export default HeaderBar;