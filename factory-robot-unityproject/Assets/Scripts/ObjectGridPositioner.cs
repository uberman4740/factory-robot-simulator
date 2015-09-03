using UnityEngine;
using System.Collections;
using System.Collections.Generic;

[ExecuteInEditMode]
public class ObjectGridPositioner : MonoBehaviour {
    public GameObject objectPrefab;

    public int nX;
    public int nZ;

    public float distX;
    public float distZ;
    public Vector3 corner;

    [SerializeField]
    private List<Object> instantiatedObjects = new List<Object>();
    public bool clearObjects;
    public bool createObjects;
    public string namePrefix;

    void Update() {
        if (createObjects 
            && objectPrefab != null
            && (instantiatedObjects == null
                || instantiatedObjects.Count == 0)) {

            instantiatedObjects = new List<Object>();
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nZ; j++) {
                    GameObject newObject = (GameObject) Instantiate(objectPrefab,
                                        new Vector3(i * distX, 0, j * distZ) + corner,
                                        Quaternion.identity);
                    newObject.name = namePrefix + i + "-" + j;
                    newObject.transform.SetParent(this.transform);
                    instantiatedObjects.Add(newObject);
                }
            }
            createObjects = false;
        }
        else if (createObjects) {
            Debug.Log("Lamps have already been created. Use Clear Lamps.");
            createObjects = false;
        }
        
        if (clearObjects) {
            foreach (Object o in instantiatedObjects) {
                DestroyImmediate(o);
            }
            instantiatedObjects.Clear();
            clearObjects = false;
        }
    }


}
