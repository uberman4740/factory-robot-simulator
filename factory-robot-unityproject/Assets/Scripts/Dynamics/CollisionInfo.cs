using UnityEngine;
using System.Collections;

public class CollisionInfo : MonoBehaviour {

	public float collisionReward;
	public bool respawnOnCollide;
	public float repel = 0.0f;

	public Vector3 respawnRangeCenter;
	public Vector3 respawnRange;

	public void Notify(Vector3 direction) {
		if (respawnOnCollide) {
			Respawn();
		} else {
//			transform.GetComponent<Rigidbody>().AddExplosionForce(repel, 
//			                                                      transform.position - 2.0f*direction, 
//			                                                      10.0f);
			transform.Translate(repel*direction, Space.World);
		}
	}

	public float GetCollisionReward() {
		return collisionReward;
	}

	private void Respawn() {
		transform.position = MathUtils.getUniformRandomVector3(
			respawnRangeCenter, respawnRange);
	}
}
