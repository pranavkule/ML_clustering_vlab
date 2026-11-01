import { useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import Aim from './components/Aim.jsx'
import Theory from './components/Theory.jsx'
import Procedure from './components/Procedure.jsx'
import Simulation from './components/Simulation.jsx'
import Quiz from './components/Quiz.jsx'

export default function App() {
  const [section, setSection] = useState('Aim')
  const pages = { Aim, Theory, Procedure, Simulation, 'Quiz Based Assessment': Quiz }
  const Page = pages[section]
  return (
    <div className="app-shell">
      <Sidebar section={section} onChange={setSection} />
      <main><div className="content"><Page /></div></main>
    </div>
  )
}
