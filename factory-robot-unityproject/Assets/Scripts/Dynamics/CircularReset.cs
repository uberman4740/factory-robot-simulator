using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class CircularReset : Resetter {

	public Transform[] objects;
	public float distanceMin;
	public float distanceMax;
	public float minAngleDiff;

	public Transform centerObject;
//		public Vector3 center;

	// Use this for initialization
	void Start () {
		if (minAngleDiff * objects.Length > 360.0f) {
			Debug.LogWarningFormat("minAngleDiff is too large for {0} objects", objects.Length);
		}

	}


	public override void ResetPosition() {
//		float[] angles = new float[objects.Length];

		int nAngleSteps = Mathf.FloorToInt(360.0f / minAngleDiff);
		List<int> angles = Enumerable.Range(0, nAngleSteps).ToList();


		for (int i = 0; i < objects.Length; i++) {

			int nextIndex = Random.Range(0, angles.Count);
			int nextAnglePosition = angles[nextIndex];
			angles.RemoveAt(nextIndex);

			float angle = nextAnglePosition * minAngleDiff;
			float distance = Random.Range (distanceMin, distanceMax);

			float x = centerObject.position.x + distance * Mathf.Cos(Mathf.Deg2Rad * angle);
			float y = centerObject.position.y;
			float z = centerObject.position.z + distance * Mathf.Sin(Mathf.Deg2Rad * angle);

			objects[i].position = new Vector3(x, y, z);
			objects[i].eulerAngles = new Vector3(0.0f, 
			                                     Random.Range(0.0f, 360.0f), 
			                                     0.0f);

			// Make CenterObject point forward
			centerObject.rotation = Quaternion.identity;
		}
	}
	
}
