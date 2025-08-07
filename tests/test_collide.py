import warp as wp
import warp.sim
import warp.sim.render

# Initialize the Warp framework
wp.init()

# --- Setup the Scene ---
builder = wp.sim.ModelBuilder()

# Define the properties and poses for two boxes
# Ensure their positions cause them to overlap for this example
box1_pos = (0.0, 1.8, 0.0)
box1_rot = (0.0, 0.0, 0.0, 1.0) # No rotation
box1_dims = (1.0, 1.0, 1.0)

box2_pos = (0.0, 1.0, 0.0)
box2_rot = (0.0, 0.0, 0.0, 1.0) # No rotation
box2_dims = (1.0, 1.0, 1.0)

body1 = builder.add_body(origin=wp.transform(box1_pos,box1_rot), armature=1.0, m=1.0)
body2 = builder.add_body(origin=wp.transform(box2_pos,box2_rot), armature=1.0, m=1.0)

# Add the shapes and assign them body IDs
# Using body IDs 0 and 1 makes it easy to identify them in the results
builder.add_shape_box(body=body1, hx=box1_dims[0] / 2.0, hy=box1_dims[1] / 2.0, hz=box1_dims[2] / 2.0)
builder.add_shape_box(body=body2, hx=box2_dims[0] / 2.0, hy=box2_dims[1] / 2.0, hz=box2_dims[2] / 2.0)

# Build the model
model = builder.finalize('cuda')

# Create a state object to hold poses and results
state = model.state()

# --- Set Body Poses and Perform Collision Check ---

# Call the collision function to populate the state with contact data
wp.sim.collide(model, state)

# --- Process and Display the Results ---

# Get the number of contacts found from the device (GPU)
num_contacts = model.rigid_contact_count.numpy()[0]

print(f"Collision check complete. Found {num_contacts} contact points.\n")

if num_contacts > 0:
    # Copy contact data from the device to the host (CPU) to print it
    # We only need to copy the number of contacts that were actually found
    contact_shape0 = model.rigid_contact_shape0.numpy()[:num_contacts]
    contact_shape1 = model.rigid_contact_shape1.numpy()[:num_contacts]
    contact_point0 = model.rigid_contact_point0.numpy()[:num_contacts]
    contact_point1 = model.rigid_contact_point1.numpy()[:num_contacts]
    contact_offset0 = model.rigid_contact_offset0.numpy()[:num_contacts]
    contact_offset1 = model.rigid_contact_offset1.numpy()[:num_contacts]
    contact_normals = model.rigid_contact_normal.numpy()[:num_contacts]
    contact_thickness = model.rigid_contact_thickness.numpy()[:num_contacts]
    #contact_pairwise_counter = model.rigid_contact_pairwise_counter.numpy()[:num_contacts]
    #contact_tids = model.contact_tids.numpy()[:num_contacts]

    print("--- Contact Details ---")
    for i in range(num_contacts):
        body_pair = (contact_shape0[i], contact_shape1[i])
        point0 = contact_point0[i]
        point1 = contact_point1[i]
        offset0 = contact_offset0[i]
        offset1 = contact_offset1[i]
        normal = contact_normals[i]
        thickness = contact_thickness[i]

        print(f"\nContact Point {i+1}:")
        print(f"  - Bodies Involved: {body_pair[0]} and {body_pair[1]}")
        print(f"  - Position0: ({point0[0]:.3f}, {point0[1]:.3f}, {point0[2]:.3f}), Position1: ({point1[0]:.3f}, {point1[1]:.3f}, {point1[2]:.3f})")
        print(f"  - Offset0: ({offset0[0]:.3f}, {offset0[1]:.3f}, {offset0[2]:.3f}), Offset1: ({offset1[0]:.3f}, {offset1[1]:.3f}, {offset1[2]:.3f})")
        print(f"  - Normal Vector: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
        print(f"  - Thickness: {abs(thickness):.4f}")
else:
    print("No collision was detected between the bodies.")