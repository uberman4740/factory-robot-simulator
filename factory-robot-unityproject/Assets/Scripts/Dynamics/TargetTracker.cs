using UnityEngine;
using System.Collections;

public class TargetTracker : MonoBehaviour {

	
	public Transform trackedTransform;
	public float pull = 0.5f;
    public float dampen = 0.9f;

	public bool smooth = true;

    private Vector3 speed;

	private Vector3 offset;

	void Start() {
		offset = transform.position - trackedTransform.position;
	}
	
	// Update is called once per frame
	void Update () {
		if (smooth) {
			Vector3 targetPosition = trackedTransform.position + offset;

			Vector3 difference = targetPosition - transform.position;

            speed += difference * pull * Time.deltaTime;
            speed *= dampen;

			transform.Translate(speed);


		} else {
			transform.position = trackedTransform.position + offset;
		}


	}
}
