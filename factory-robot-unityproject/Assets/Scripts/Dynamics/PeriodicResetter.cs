using UnityEngine;
using System.Collections;

public class PeriodicResetter : MonoBehaviour {

	public int resetPeriod = 100;

	public Transform[] objects;
	public Vector2[] directionRange;
	public Vector2[] positionRangeMin;
	public Vector2[] positionRangeMax;

//	private Vector3[] positions;


	private int counter = 0;
	private bool hasBeenReset = false;

	void Start () {
		if (directionRange.Length != objects.Length) {
			Debug.LogWarning("resetDirection.Length != objects.Length");
		}
		if (directionRange.Length != positionRangeMin.Length) {
			Debug.LogWarning("resetDirection.Length != positionRangeMin.Length");
		}
		if (directionRange.Length != positionRangeMax.Length) {
			Debug.LogWarning("resetDirection.Length != positionRangeMax.Length");
		}

//		positions = new Vector3[objects.Length];
//		for (int i = 0; i < objects.Length; i++) {
//			positions[i] = objects[i].position;
//		}
	}
	
	void Update () {
		counter++;

		if ((counter % resetPeriod) == 0) {
			Reset();
		}
	}

	private void Reset() {
		hasBeenReset = true;
		for (int i = 0; i < objects.Length; i++) {
//			objects[i].position = positions[i];
			objects[i].position = new Vector3(Random.Range(positionRangeMin[i].x, positionRangeMax[i].x),
			                                  0,
			                                  Random.Range(positionRangeMin[i].y, positionRangeMax[i].y));
			objects[i].eulerAngles = new Vector3(0, 
			                                     Random.Range(directionRange[i].x, 
			                                     			  directionRange[i].y), 
			                                     0);

		}
	}

//	public bool HasBeenReset() {
//		return (hasBeenReset = false);
//	}
}
