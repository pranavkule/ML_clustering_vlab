import { useEffect, useState } from 'react'
import { api } from '../services/api.js'

export default function Quiz() {
  const [questions, setQuestions] = useState([])
  const [answers, setAnswers] = useState(Array(10).fill(null))
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => { api.quiz().then((data) => setQuestions(data.questions)).catch((e) => setError(e.message)) }, [])

  const submit = async () => {
    if (answers.some((answer) => answer === null)) { setError('Please answer all ten questions before submitting.'); return }
    try { setResult(await api.submitQuiz(answers)); setError('') } catch (e) { setError(e.message) }
  }
  const restart = () => { setAnswers(Array(10).fill(null)); setResult(null); setError('') }
  return (
    <>
      <h1>Quiz Based Assessment</h1><div className="rule" />
      {error && <p className="notice error">{error}</p>}
      {!result ? questions.map((item, index) => (
        <fieldset className="quiz-question" key={index}>
          <legend>{index + 1}. {item.question}</legend>
          {item.options.map((option, optionIndex) => (
            <label className="quiz-option" key={option}><input type="radio" name={`q-${index}`} checked={answers[index] === optionIndex} onChange={() => setAnswers((current) => current.map((value, i) => i === index ? optionIndex : value))} />{option}</label>
          ))}
        </fieldset>
      )) : (
        <>
          <div className="metrics"><div><span>Score</span><strong>{result.score} / 10</strong></div><div><span>Percentage</span><strong>{result.percentage}%</strong></div><div><span>Grade</span><strong>{result.grade}</strong></div></div>
          {result.review.map((item, index) => (
            <section className="review" key={index}>
              <strong>{index + 1}. {item.question}</strong>
              <p className={result.flags[index] ? 'correct' : 'incorrect'}>{result.flags[index] ? 'Correct' : 'Incorrect'}</p>
              {!result.flags[index] && <p>Correct answer: {item.options[item.answer_index]}</p>}
              <p>{item.explanation}</p>
            </section>
          ))}
        </>
      )}
      <div className="button-row">
        {result ? <button className="primary" type="button" onClick={restart}>Restart Quiz</button> : <button className="primary" type="button" onClick={submit}>Submit Quiz</button>}
      </div>
    </>
  )
}
