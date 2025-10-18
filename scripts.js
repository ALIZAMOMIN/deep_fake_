const themeToggle = document.getElementById('themeToggle');
const root = document.documentElement;

if(localStorage.getItem('vf_theme'))
  root.setAttribute('data-theme', localStorage.getItem('vf_theme'));

themeToggle.addEventListener('click', () => {
  const cur = root.getAttribute('data-theme') || 'dark';
  const next = cur === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  localStorage.setItem('vf_theme', next);
});

function openAuthContainer() {
  const container = document.getElementById('authContainer');
  container.style.display = 'flex';
  document.body.classList.add('auth-active');
  switchAuth('login');
}

function closeAuthContainer() {
  const container = document.getElementById('authContainer');
  container.style.display = 'none';
  document.body.classList.remove('auth-active');
}

function switchAuth(mode) {
  const authTitle = document.getElementById('authTitle');
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');

  if(mode === 'signup') {
    authTitle.textContent = 'Sign Up';
    loginForm.style.display = 'none';
    signupForm.style.display = 'block';
  } else {
    authTitle.textContent = 'Login';
    loginForm.style.display = 'block';
    signupForm.style.display = 'none';
  }
}

function fakeLogin() {
  alert('Login is mocked for demo. Welcome back!');
  closeAuthContainer();
}

function fakeSignup() {
  alert('Account created (demo).');
  closeAuthContainer();
}

/* 
Insert your other existing JS code here as needed:
- Particle effect
- Demo file upload & scan simulation
- Report submission 
- Trust score handling 
Exclude chatbot or other removed components.
*/
