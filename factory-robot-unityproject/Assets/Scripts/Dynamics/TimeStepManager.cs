using UnityEngine;
using System.Collections;

public class TimeStepManager : MonoBehaviour {

	/* Simulated time per frame, in seconds. */
	public float deltaTime;

	public enum State { Advance, Receive }

	/* Timeout control */
	private float lastSendTime;
	public float timeoutPeriod;

	public SensorConverter sensorConverter;
	public SocketSender socketSender;
	public InputListener inputListener;

	private State _state;
	public State state {
		get {
			return _state;
		}
		set {
			_state = value;
		}
	}


	void Update() {
		if (_state == State.Advance) {
			SendFrame();
			_state = State.Receive;
		}

		if (_state == State.Receive) {
			if (inputListener.ReceivedBytes()) {
				if (inputListener.currentFrameCounter != sensorConverter.frameCounter) {
					Debug.LogFormat ("Received wrong frame counter ({0} instead of {1}).",
					                 inputListener.currentFrameCounter, 
					                 sensorConverter.frameCounter);
				} else {
					_state = State.Advance;
					sensorConverter.AdvanceFrameCounter();
				}
			}

			if (IsTimedOut()) {
				Debug.LogFormat("{1,5} Timeout. sensorConverter.frameCounter={0}", sensorConverter.frameCounter, Time.realtimeSinceStartup);
				SendFrame();
			}
		}
	}

	public void SendFrame() {
		byte[][] frame = sensorConverter.GetCurrentFragments();
		socketSender.SendFrame(frame);
		SetSentFrame();
	}

	public void SetSentFrame() {
		// set timeout times
		lastSendTime = Time.realtimeSinceStartup;
		_state = State.Receive;
	}

	private bool IsTimedOut() {
		return _state == State.Receive 
			&& Time.realtimeSinceStartup - lastSendTime > timeoutPeriod;
	}
}
