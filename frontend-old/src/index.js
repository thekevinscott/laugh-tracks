import WaveSurfer from 'wavesurfer.js';
import TimelinePlugin from 'wavesurfer.js/src/plugin/timeline.js';
// import MinimapPlugin from 'wavesurfer.js/dist/plugin/wavesurfer.minimap.min.js';
const audio = require('./test.wav');

const rand = () => {
  return Math.round(Math.random() * 10000);
}

const createDiv = (id, style) => {
  const div = document.createElement('div');
  div.id = `container-${id}`;
  div.style = style;
  return div;
};

alert('1');

// const createSpan = (id, style) => {
//   const span = document.createElement('span');
//   span.class = `span-${id}`;
//   span.style = style;
//   return span;
// };

const createSlider = (id, style) => {
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.id = `container-${id}`;
  slider.min = 1;
  slider.max = 200;
  slider.value = 1;
  slider.style = style;
  return slider;
};

const createButton = (id) => {
  const button = document.createElement('button');
  button.id = `button-${id}`;
  button.innerHTML = 'play';
  button.disabled = true;
  return button;
}

const main = (target, annotations) => {
  const id = `${rand()}`;
  const container = `#container-${id}`;
  const div = createDiv(id);
  const timeline = createDiv(`#timeline-${id}`);
  const controls = createDiv(`#controls-${id}`, 'display: flex;');
  const annotation = createDiv(`#controls-${id}`, 'display: flex');
  const button = createButton(id);
  const slider = createSlider(id, 'flex: 1');
  target.append(div);
  target.append(timeline);
  target.append(annotation);
  target.append(controls);
  controls.append(button);
  controls.append(slider);
  const wavesurfer = WaveSurfer.create({
    container,
    waveColor: 'violet',
    plugins: [
      TimelinePlugin.create({
        container: timeline,
        markers: annotations,
      }),
    ]
  });

  slider.oninput = () => {
    const zoomLevel = Number(slider.value);
    console.log('hi', zoomLevel);
    wavesurfer.zoom(zoomLevel);
  };
  wavesurfer.load(audio);
  wavesurfer.on('ready', () => {
    button.disabled = false;

    button.addEventListener('click', () => {
      wavesurfer.playPause();
    });
    wavesurfer.on('pause', () => {
      button.innerHTML = 'play';
    });
    wavesurfer.on('play', () => {
      button.innerHTML = 'pause';
    });
  });
}

main(document.body, [{
  label: 'a',
  timestamp: 0.5,
}, {
  label: 'b',
  timestamp: 1.5,
}]);
