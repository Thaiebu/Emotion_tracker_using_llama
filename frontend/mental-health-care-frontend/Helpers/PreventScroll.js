export const numberInputOnWheelPreventChange = (e) => {
  e.target.blur();
  setTimeout(() => {
    e.target.focus();
  }, 0);
};

// Debounce function to limit the rate at which the handleResize function is invoked
export const debounce = (func, wait) => {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
};