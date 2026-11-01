export default function Sidebar({ section, onChange }) {
  const sections = ['Aim', 'Theory', 'Procedure', 'Simulation', 'Quiz Based Assessment']
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">Clustering Lab</div>
      <nav aria-label="Laboratory sections">
        {sections.map((item) => (
          <button
            key={item}
            type="button"
            className={section === item ? 'nav-item active' : 'nav-item'}
            onClick={() => onChange(item)}
          >
            {item}
          </button>
        ))}
      </nav>
    </aside>
  )
}
