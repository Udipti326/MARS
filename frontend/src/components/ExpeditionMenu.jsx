import { useEffect, useRef, useState } from "react";

function ExpeditionMenu({ onView, onDelete, onChat }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handleClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div className="menu-wrap" ref={ref}>
      <button className="icon-button menu-trigger" onClick={() => setOpen((v) => !v)}>
        ☰
      </button>

      {open && (
        <div className="menu-dropdown">
          <button
            className="menu-item"
            onClick={() => {
              setOpen(false);
              onView?.();
            }}
          >
            View
          </button>
          <button
            className="menu-item"
            onClick={() => {
              setOpen(false);
              onChat?.();
            }}
          >
            Chat
          </button>
          <button
            className="menu-item menu-danger"
            onClick={() => {
              setOpen(false);
              onDelete?.();
            }}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

export default ExpeditionMenu;