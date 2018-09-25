import React from 'react';
import ReactDOM from 'react-dom';
import Audio from './Audio';

const render = (target, audio, markers) => ReactDOM.render((
  <Audio
    audio={audio}
    markers={markers}
  />
), target);

export default render;
