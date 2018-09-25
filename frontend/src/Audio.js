import WaveSurfer from 'wavesurfer.js';
import TimelinePlugin from 'wavesurfer.js/src/plugin/timeline.js';
import React, { Component } from 'react';

class App extends Component {
  shouldComponentUpdate() {
    return false;
  }

  componentDidMount() {
    const wavesurfer = WaveSurfer.create({
      container: "#app",
      waveColor: 'violet',
      plugins: [
        // TimelinePlugin.create({
        //   // container: timeline,
        //   // markers: annotations,
        // }),
      ]
    });
    wavesurfer.load('/test.wav');
  }

  render() {
    return (
      <div id="app" />
    );
  }
}

export default App;
