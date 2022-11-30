from pybullet_utils import bullet_client


def loadOBJ(
    p: bullet_client.BulletClient,
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
        baseOrientation = p.getQuaternionFromEuler(baseOrientation)

    if visualId == -1:
        visualId = obj_visual(fileName, meshScale)

    return p.createMultiBody(
        baseMass=baseMass,
        baseVisualShapeIndex=int(visualId),
        baseCollisionShapeIndex=int(collisionId),
        basePosition=basePosition,
        baseOrientation=baseOrientation,
    )


def obj_visual(p: bullet_client.BulletClient, fileName, meshScale=[1.0, 1.0, 1.0]):
    return p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=fileName,
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0.0, 0.0, 0.0],
        meshScale=meshScale,
    )


def obj_collision(p: bullet_client.BulletClient, fileName, meshScale=[1.0, 1.0, 1.0]):
    return p.createCollisionShape(
        shapeType=p.GEOM_MESH, fileName=fileName, meshScale=meshScale
    )
