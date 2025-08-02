import axios from 'axios'

// Configure axios defaults
// Use relative URLs so Vite proxy handles them
axios.defaults.headers.common['Content-Type'] = 'application/json'

export default axios