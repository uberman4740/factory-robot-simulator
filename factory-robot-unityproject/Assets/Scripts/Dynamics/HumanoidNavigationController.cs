using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityStandardAssets.Characters.ThirdPerson;

class NavTarget {
    public Transform transform;
    public bool reserved;

    public NavTarget(Transform transform) {
        this.transform = transform;
        this.reserved = false;
    }
}

class NavCharacter {
    public AICharacterControl aiCharacterControl;
    public float nextChangeTime;
    public NavTarget currentTarget;

    public NavCharacter(AICharacterControl aiCharacterControl, NavTarget currentTarget) {
        this.aiCharacterControl = aiCharacterControl;
        this.nextChangeTime = 0.0f;
        this.currentTarget = currentTarget;
    }
}

public class HumanoidNavigationController : MonoBehaviour {

    /** Number of seconds until a character changes his target */
    public float positionUpdateIntervalRangeMin = 5.0f;
    public float positionUpdateIntervalRangeMax = 25.0f;


    /** Number of seconds between two update calls */
    public float controllerUpdateInterval = 0.5f;
    private float lastUpdate;

    /** Distance from target such that it is considered reached */
    public float distanceThreshold = 0.2f;

    /** Direction the character should turn towards when target is reached */
    public Vector3 facingDirection = new Vector3(-1.0f, 0, 0);


    public AICharacterControl[] characterControllers;
    public Transform[] targets;

    [SerializeField]
    private NavTarget[] navTargets;
    private NavCharacter[] navCharacters;



	// Use this for initialization
	void Start () {
        navTargets = new NavTarget[targets.Length];
        for (int i = 0; i < targets.Length; i++) {
            navTargets[i] = new NavTarget(targets[i]);
        }

        navCharacters = new NavCharacter[characterControllers.Length];
        for (int i = 0; i < characterControllers.Length; i++) {
            var target = SampleAvailableTarget();
            target.reserved = true;
            navCharacters[i] = new NavCharacter(characterControllers[i], target);
        }

        lastUpdate = Time.time;
	}
	
	// Update is called once per frame
	void Update () {
        if (Time.time - lastUpdate > controllerUpdateInterval) {
            UpdateControllers();            
            lastUpdate = Time.time;
        }
	}

    private void UpdateControllers() {
        for (int i = 0; i < navCharacters.Length; i++) {
            if (Time.time > navCharacters[i].nextChangeTime) {

                NavTarget nextTarget = SampleAvailableTarget();
                navCharacters[i].aiCharacterControl.SetTarget(nextTarget.transform);

                navCharacters[i].currentTarget.reserved = false;
                nextTarget.reserved = true;
                navCharacters[i].currentTarget = nextTarget;

                navCharacters[i].nextChangeTime = Time.time + Random.Range(
                        positionUpdateIntervalRangeMin, 
                        positionUpdateIntervalRangeMax);
            } else if (navCharacters[i].aiCharacterControl.agent.remainingDistance < distanceThreshold) {
                navCharacters[i].aiCharacterControl.SetTarget(null);
                navCharacters[i].aiCharacterControl.transform.rotation = Quaternion.LookRotation(facingDirection);
            }
        }
    }

    private NavTarget SampleAvailableTarget() {
        List<NavTarget> availableTargets = new List<NavTarget>();
        for (int i = 0; i < navTargets.Length; i++) {
            if (!navTargets[i].reserved) {
                availableTargets.Add(navTargets[i]);
            }
        }


        if (availableTargets.Count == 0) {
            Debug.LogWarning("No available Targets for NavigationController!");
        }

        return availableTargets[Random.Range(0, availableTargets.Count)];
    }
}
