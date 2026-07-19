from credential_store.store import CredentialStore


def capture_and_clear_camera_credentials(project, store: CredentialStore) -> int:

    # The one place a plaintext camera password is allowed to exist
    # transiently in memory (freshly typed in the Property Panel, or
    # just loaded from a legacy pre-credential_ref project file) gets
    # captured into `store` and cleared from the in-memory model here.
    # Same operation either way -- there is no meaningful difference
    # between "a user just typed a password" and "an old project file
    # still has one loaded" from this function's point of view, so one
    # function serves both serialization/serializer.py's save() and
    # load() hooks.
    #
    # Walks building -> floors -> cameras the same way
    # camera_manager.manager.CameraManager.discover_cameras() does --
    # deliberately not importing CameraManager itself here to keep
    # this module free of any camera_manager/multi_camera_fusion
    # dependency (only models.Project's own shape is needed).
    #
    # Returns how many cameras were migrated/captured, for tests and
    # logging.

    if project is None or project.building is None:
        return 0

    migrated = 0

    for floor in project.building.ordered_floors():
        for camera in floor.cameras:

            connection = camera.connection

            if not connection.password:
                continue

            reference_id = connection.credential_ref or camera.id

            store.save_credential(reference_id, connection.password)

            connection.credential_ref = reference_id
            connection.password = ""

            migrated += 1

    return migrated
