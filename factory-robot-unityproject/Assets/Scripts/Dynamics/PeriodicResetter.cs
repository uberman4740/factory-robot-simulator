using UnityEngine;
using System.Collections;

public class PeriodicResetter : MonoBehaviour {

	public int resetPeriod = 100;

	public Resetter resetter;

	private int counter = 0;

	void Start() {
	}
	
	void Update () {

		if ((counter % resetPeriod) == 0) {
			resetter.ResetPosition();
		}

		counter++;
	}

	
}
