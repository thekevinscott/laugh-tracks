import render from './App';
const markers = [{
  label: 'foo',
  timestamp: 2,
}, {
  label: 'bar',
  timestamp: 5,
}];

render(document.getElementById('root'), '/test.wav', markers);
