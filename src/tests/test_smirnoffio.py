import pytest

has_openff_toolkit = True
try:
    from openff.toolkit.typing.engines.smirnoff import ForceField
    from openff.toolkit.typing.engines.smirnoff.parameters import VirtualSiteHandler
    from openff.units import unit
except ModuleNotFoundError:
    has_openff_toolkit = False

from forcebalance.smirnoffio import assign_openff_parameter, select_virtual_site_parameter


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def _build_virtual_site_force_field(include_incomplete=False):
    force_field = ForceField()
    vsite_handler = VirtualSiteHandler(version=0.3)

    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#17:2]",
            "name": "EP1",
            "type": "BondCharge",
            "distance": 0.10 * unit.nanometers,
            "match": "all_permutations",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )
    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#17:2]",
            "name": "EP2",
            "type": "BondCharge",
            "distance": 0.20 * unit.nanometers,
            "match": "all_permutations",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )

    if include_incomplete:
        vsite_handler.add_parameter(
            {
                "smirks": "[#1:1]-[#17:2]",
                "name": "EP3",
                "type": "BondCharge",
                "distance": 0.30 * unit.nanometers,
                "match": "all_permutations",
                "charge_increment": [
                    0.0 * unit.elementary_charge,
                    0.0 * unit.elementary_charge,
                ],
            }
        )
        vsite_handler.parameters[-1].match = None

    force_field.register_parameter_handler(vsite_handler)
    return force_field


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def test_select_virtual_site_parameter_requires_non_none_identifiers():
    force_field = _build_virtual_site_force_field()

    with pytest.raises(KeyError, match="requires non-None identifiers"):
        select_virtual_site_parameter(
            parameters=force_field.get_parameter_handler("VirtualSites").parameters,
            smirks="[#1:1]-[#17:2]",
            virtual_site_type=None,
            virtual_site_name="EP1",
            virtual_site_match="all_permutations",
            error_context="unit test",
        )


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def test_select_virtual_site_parameter_selects_unique_match():
    force_field = _build_virtual_site_force_field(include_incomplete=True)

    parameter = select_virtual_site_parameter(
        parameters=force_field.get_parameter_handler("VirtualSites").parameters,
        smirks="[#1:1]-[#17:2]",
        virtual_site_type="BondCharge",
        virtual_site_name="EP1",
        virtual_site_match="all_permutations",
    )

    assert parameter.name == "EP1"


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def test_select_virtual_site_parameter_raises_for_ambiguous_match():
    force_field = ForceField()
    vsite_handler = force_field.get_parameter_handler("VirtualSites")
    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "name": f"LP",
            "type": "DivalentLonePair",
            "distance": -0.0106 * unit.nanometers,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )
    duplicate_parameter_list = list(vsite_handler.parameters) * 3

    with pytest.raises(KeyError, match="Multiple VirtualSites parameters matched"):
        select_virtual_site_parameter(
            parameters=duplicate_parameter_list,
            smirks="[#1:1]-[#8X2H2+0:2]-[#1:3]",
            virtual_site_type="DivalentLonePair",
            virtual_site_name="LP",
            virtual_site_match="once",
            error_context="ambiguous unit test",
        )


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def test_assign_openff_parameter_virtual_site_pid_disambiguation():
    force_field = ForceField()
    vsite_handler = force_field.get_parameter_handler("VirtualSites")

    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#17:2]",
            "name": "EP1",
            "type": "BondCharge",
            "distance": 0.10 * unit.nanometers,
            "match": "all_permutations",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )
    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#17:2]",
            "name": "EP2",
            "type": "BondCharge",
            "distance": 0.20 * unit.nanometers,
            "match": "all_permutations",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )

    assign_openff_parameter(
        force_field,
        0.15,
        "VirtualSites/VirtualSite/distance/[#1:1]-[#17:2]/BondCharge/EP2/all_permutations",
    )

    ep2 = [
        parameter
        for parameter in force_field.get_parameter_handler("VirtualSites").parameters
        if parameter.name == "EP2"
    ][0]

    assert pytest.approx(ep2.distance.to(unit.nanometer).magnitude) == 0.15


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def test_select_virtual_site_parameter_divalent_lone_pair_disambiguation():
    force_field = ForceField()
    vsite_handler = force_field.get_parameter_handler("VirtualSites")

    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "name": "LP1",
            "type": "DivalentLonePair",
            "distance": -0.0106 * unit.nanometers,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )
    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "name": "LP2",
            "type": "DivalentLonePair",
            "distance": -0.0200 * unit.nanometers,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )

    parameter = select_virtual_site_parameter(
        parameters=force_field.get_parameter_handler("VirtualSites").parameters,
        smirks="[#1:1]-[#8X2H2+0:2]-[#1:3]",
        virtual_site_type="DivalentLonePair",
        virtual_site_name="LP2",
        virtual_site_match="once",
    )

    assert parameter.name == "LP2"


@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit module not found"
)
def test_assign_openff_parameter_divalent_lone_pair_pid_disambiguation():
    force_field = ForceField()
    vsite_handler = force_field.get_parameter_handler("VirtualSites")

    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "name": "LP1",
            "type": "DivalentLonePair",
            "distance": -0.0106 * unit.nanometers,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )
    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1]-[#8X2H2+0:2]-[#1:3]",
            "name": "LP2",
            "type": "DivalentLonePair",
            "distance": -0.0200 * unit.nanometers,
            "outOfPlaneAngle": 0.0 * unit.degrees,
            "match": "once",
            "charge_increment": [
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
                0.0 * unit.elementary_charge,
            ],
        }
    )

    with pytest.raises(ValueError, match="VirtualSites parameter ID must include type/name/match"):
        assign_openff_parameter(
            force_field,
            -0.015,
            "VirtualSites/VirtualSite/distance/[#1:1]-[#8X2H2+0:2]-[#1:3]",
        )

    assign_openff_parameter(
        force_field,
        -0.015,
        "VirtualSites/VirtualSite/distance/[#1:1]-[#8X2H2+0:2]-[#1:3]/DivalentLonePair/LP2/once",
    )

    lp2 = [
        parameter
        for parameter in force_field.get_parameter_handler("VirtualSites").parameters
        if parameter.name == "LP2"
    ][0]

    assert pytest.approx(lp2.distance.to(unit.nanometer).magnitude) == -0.015
