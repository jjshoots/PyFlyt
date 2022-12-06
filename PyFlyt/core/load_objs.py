from .aviary import Aviary


def loadOBJ(
    env: Aviary,
    fileName="null",
    visualId=-1,
    collisionId=-1,
    baseMass=0.0,
    meshScale=[1.0, 1.0, 1.0],
    basePosition=[0.0, 0.0, 0.0],
    baseOrientation=[0.0, 0.0, 0.0],
):
    """
    loads in an object via either the fileName or meshId, meshId takes precedence
    """
    if len(baseOrientation) == 3:
        baseOrientation = env.getQuaternionFromEuler(baseOrientation)

    if visualId == -1:
        visualId = obj_visual(fileName, meshScale)

    body_id = env.createMultiBody(
        baseMass=baseMass,
        baseVisualShapeIndex=int(visualId),
        baseCollisionShapeIndex=int(collisionId),
        basePosition=basePosition,
        baseOrientation=baseOrientation,
    )

    env.register_all_new_bodies()

    return body_id


def obj_visual(env: Aviary, fileName, meshScale=[1.0, 1.0, 1.0]):
    return env.createVisualShape(
        shapeType=env.GEOM_MESH,
        fileName=fileName,
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.0, 0.0, 0.0],
        meshScale=meshScale,
    )


def obj_collision(env: Aviary, fileName, meshScale=[1.0, 1.0, 1.0]):
    return env.createCollisionShape(
        shapeType=env.GEOM_MESH, fileName=fileName, meshScale=meshScale
    )
