import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link href="/" className="brand">
          <span style={{ color: "#38bdf8" }}>❖</span>
          <span>ClusterOps BFF</span>
        </Link>
        <div className="nav-links">
          <Link href="/" className="nav-link">
            Architecture Guide
          </Link>
          <Link href="/dashboard" className="nav-link" style={{ fontWeight: 600, color: "#38bdf8" }}>
            Live Dashboard (RSC + Streaming)
          </Link>
        </div>
      </div>
    </nav>
  );
}
