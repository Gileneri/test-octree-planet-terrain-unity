using UnityEngine;

public class LookAround : MonoBehaviour
{
    public new Transform camera;

    [Header("Movement")]
    [Tooltip("Normal movement speed (units per second).")]
    public float normalSpeed = 20f;

    [Tooltip("Fast movement speed when holding Fire3 (Left Shift).")]
    public float fastSpeed = 200f;

    [Tooltip("Mouse look sensitivity (degrees per tick).")]
    public float mouseSensitivity = 2f;

    private void Start()
    {
        Cursor.lockState = CursorLockMode.Locked;
    }

    private void Update()
    {
        // ── Movement ─────────────────────────────────────────────────────
        float forward = Input.GetAxis("Vertical");
        float right = Input.GetAxis("Horizontal");
        float up = Input.GetAxis("UpDown");

        float speed = Input.GetButton("Fire3") ? fastSpeed : normalSpeed;

        // Multiply by Time.deltaTime so speed is frame-rate independent.
        // Previously the movement was per-frame, so a meshing spike would
        // cause the player to move slower during heavy LOD transitions.
        transform.position += camera.forward * forward * speed * Time.deltaTime;
        transform.position += transform.up * up * speed * Time.deltaTime;
        transform.position += transform.right * right * speed * Time.deltaTime;

        // ── Look ──────────────────────────────────────────────────────────
        float rotateY = Input.GetAxis("Mouse X");
        float rotateX = Input.GetAxis("Mouse Y");

        transform.Rotate(Vector3.up, rotateY * mouseSensitivity, Space.World);
        camera.Rotate(Vector3.right, -rotateX * mouseSensitivity);
    }
}