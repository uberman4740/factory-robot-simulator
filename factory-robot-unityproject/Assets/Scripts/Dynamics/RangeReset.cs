using UnityEngine;
using System.Collections;

public class RangeReset : Resetter {

	public Transform[] objects;
	public Vector2[] directionRange;
	public Vector2[] positionRangeMin;
	public Vector2[] positionRangeMax;

	
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
	}
	
	public override void ResetPosition() {
		for (int i = 0; i < objects.Length; i++) {
			objects[i].position = new Vector3(Random.Range(positionRangeMin[i].x, positionRangeMax[i].x),
			                                  0,
			                                  Random.Range(positionRangeMin[i].y, positionRangeMax[i].y));
			objects[i].eulerAngles = new Vector3(0, 
			                                     Random.Range(directionRange[i].x, 
			             						 directionRange[i].y), 
			                                     0);
			
		}
	}
}
