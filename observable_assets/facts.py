from typing import FrozenSet, Mapping, Sequence, Tuple

from observable_assets.models import AssetObservation, ObservableAssetSnapshot, ObservationStatus


def compute_asset_occupancy_snapshot(
    asset_ids_by_type: Mapping[str, Sequence[str]],
    occupant_ids_by_asset: Mapping[str, Tuple[str, ...]],
    covered_asset_ids: FrozenSet[str],
    timestamp: float,
) -> ObservableAssetSnapshot:

    # Observable Asset Perception Framework milestone -- a PURE grouping
    # function, exactly mirroring live_occupants.occupancy.
    # compute_occupancy_facts()'s own "does NOT decide who counts, does
    # NOT re-scan raw occupants" convention: `occupant_ids_by_asset` is
    # taken directly from an already-computed live_occupants.occupancy.
    # OccupancyFacts-shaped grouping, and `covered_asset_ids` is taken
    # directly from camera_calibration.asset_lookup.covered_asset_ids()
    # (the geometrically-derived coverage answer) -- this function only
    # combines the two into the OBSERVED-vs-UNKNOWN distinction Phase 18
    # of the prior audit required.
    #
    # `asset_ids_by_type`: every known asset id for each asset type
    # (e.g. {"Stair": [...]}), grouped by type purely so a caller with
    # several registered asset types can supply them all in ONE call
    # without this function needing to know which types exist (it never
    # branches on asset_type at all -- see the loop below). A
    # genuinely-covered asset with zero current occupants still gets an
    # explicit OBSERVED/empty entry, never silently absent --
    # ObservableAssetSnapshot.observation_for()'s own default already
    # handles "entirely absent" as UNKNOWN, but being explicit here is
    # what lets a caller enumerate "every asset we know about," not just
    # the ones that happened to have someone on them.

    observations = {}

    for asset_type, asset_ids in asset_ids_by_type.items():

        for asset_id in asset_ids:

            occupant_ids = occupant_ids_by_asset.get(asset_id, ())

            if asset_id in covered_asset_ids:

                observations[asset_id] = AssetObservation(
                    asset_id=asset_id, asset_type=asset_type,
                    status=ObservationStatus.OBSERVED,
                    occupant_ids=tuple(occupant_ids),
                    timestamp=timestamp,
                )

            else:

                # UNKNOWN -- no honest basis to report an occupancy
                # count, regardless of whether occupant_ids_by_asset
                # happens to (impossibly, absent a bug elsewhere) carry
                # an entry for this id. occupant_ids stays deliberately
                # empty -- see AssetObservation.occupant_ids' own
                # docstring.
                observations[asset_id] = AssetObservation(
                    asset_id=asset_id, asset_type=asset_type,
                    status=ObservationStatus.UNKNOWN,
                    timestamp=timestamp,
                    provenance="no observable region authored, or no calibrated camera on its floor this cycle",
                )

    return ObservableAssetSnapshot(timestamp=timestamp, observations=observations)
