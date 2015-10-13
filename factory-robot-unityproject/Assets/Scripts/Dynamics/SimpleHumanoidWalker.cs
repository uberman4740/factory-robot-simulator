using UnityEngine;
using System.Collections;
//using UnityStandardAssets.Characters.ThirdPerson;

public class SimpleHumanoidWalker : MonoBehaviour {

//	public ThirdPersonCharacter character { get; private set; } // the character we are controlling

	public TimeStepManager timeStepManager;
	private Animator animator;

	void Start() {
//		character = GetComponent<ThirdPersonCharacter>();
		animator = GetComponent<Animator>();
	}
	
	// Update is called once per frame
	void Update () {
		if (timeStepManager.state != TimeStepManager.State.Advance) {
			return;
		}
//		character.Move(0.3f*Vector3.right, false, false);
		float forwardAmount = 0.1f;
		animator.SetFloat("Forward", forwardAmount, 0.1f, timeStepManager.deltaTime);
//		animator.SetF
	}
}
