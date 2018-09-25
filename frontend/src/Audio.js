import WaveSurfer from 'wavesurfer.js';
import TimelinePlugin from './timeline';
import React, { Component } from 'react';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      id: `a-${Math.round(Math.random() * 10000)}`,
      ready: false,
      playing: false,
    };
  }

  // shouldComponentUpdate() {
    // return false;
  // }

  componentDidMount() {
    const wavesurfer = WaveSurfer.create({
      container: `#${this.state.id}`,
      waveColor: 'violet',
      plugins: [
        TimelinePlugin.create({
          container: `#timeline-${this.state.id}`,
          timeInterval: 1,
          markers: this.props.markers || [],
        }),
      ]
    });
    wavesurfer.load(this.props.audio);
    wavesurfer.on('ready', () => {
      this.setState({
        ready: true,
      });

      wavesurfer.on('pause', () => {
        this.setState({
          playing: false,
        });
      });
      wavesurfer.on('play', () => {
        this.setState({
          playing: true,
        });
      });
    });
    this.wavesurfer = wavesurfer;
  }

  playPause = () => {
    this.wavesurfer.playPause();
  }

  zoom = (e) => {
    const zoomLevel = Number(e.target.value);
    this.wavesurfer.zoom(zoomLevel);
  }

  render() {
    return (
      <div>
        <div id={this.state.id} />
        <div id={`timeline-${this.state.id}`} />
        <div style={{ display: 'flex' }}>
          <button
            disabled={!this.state.ready}
            onClick={() => this.playPause()}
          >
            {this.state.playing ? 'pause' : 'play'}
          </button>
          <input
            style={{ flex: 1 }}
            type="range"
            defaultValue="1"
            min="1"
            max="200"
            onChange={this.zoom}
          />
        </div>
      </div>
    );
  }
}

export default App;
