using UnityEngine;
using System.Collections;

public class AgentController : MonoBehaviour {

    public float maxSpeed = 3.0f;
    public float turnSpeed = 180.0f;
    public float steeringAcceleration = 4.0f;

    public float acceleration = 1.0f;
    public float deceleration = 2.0f;

    private float currentSpeed = 0.0f;
    private float currentAngle = 0.0f;

    private float steeringAmount = 0.0f;
 

	// Update is called once per frame
	void Update () {

        if (Input.GetKey(KeyCode.UpArrow)) {
            if (currentSpeed < maxSpeed) {
                currentSpeed += acceleration * Time.deltaTime;
            } else {
                currentSpeed = maxSpeed;
            }
            if (currentSpeed < 0) {
                currentSpeed -= Mathf.Sign(currentSpeed) * deceleration * Time.deltaTime;
            }
        } else if (Input.GetKey(KeyCode.DownArrow)) {
            if (currentSpeed > -maxSpeed) {
                currentSpeed -= acceleration * Time.deltaTime;
            } else {
                currentSpeed = -maxSpeed;
            }
            if (currentSpeed > 0) {
                currentSpeed -= Mathf.Sign(currentSpeed) * deceleration * Time.deltaTime;
            }
        } else {
            float change = Mathf.Sign(currentSpeed) * deceleration * Time.deltaTime;
            if (Mathf.Abs(currentSpeed) > change) {
                currentSpeed -= Mathf.Sign(currentSpeed) * deceleration * Time.deltaTime;
            } else {
                currentSpeed = 0.0f;
            }    
        }

        if (Input.GetKey(KeyCode.LeftArrow)) {
            if (steeringAmount > -turnSpeed) {
                steeringAmount -= steeringAcceleration * Time.deltaTime;
            } else {
                steeringAmount = -turnSpeed;
            }
        } else if (Input.GetKey(KeyCode.RightArrow)) {
            if (steeringAmount < turnSpeed) {
                steeringAmount += steeringAcceleration * Time.deltaTime;
            } else {
                steeringAmount = turnSpeed;
            }
        } else {
            steeringAmount *= 0.95f;
        }


        currentAngle += Time.deltaTime * steeringAmount * currentSpeed;


        transform.eulerAngles = new Vector3(0, currentAngle, 0);
        transform.Translate(Vector3.forward * currentSpeed * Time.smoothDeltaTime);

        //transform.Translate(
        //    new Vector3(Mathf.Sin(Mathf.Deg2Rad*currentAngle), 
        //    0, 
        //    Mathf.Cos(Mathf.Deg2Rad*currentAngle)) * currentSpeed * Time.deltaTime);

	    
	}
}
