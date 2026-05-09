from __future__ import annotations

import json
from dataclasses import asdict
from decimal import Decimal

from .core import Atom, NetworkConfig, PolymerNetwork
from .params import SimulationConfig


def render_lammps_data(network: PolymerNetwork) -> str:
    lines: list[str] = []
    lines.append("LAMMPS data file for polymer network via polymer_network generator")
    lines.append("")
    lines.append(f"{len(network.atoms)} atoms")
    lines.append("3 atom types")
    lines.append(f"{len(network.bonds)} bonds")
    lines.append("4 bond types")
    lines.append(f"{len(network.atoms)} ellipsoids")
    lines.append("")
    box = _fmt(network.box_length)
    lines.append(f"0 {box} xlo xhi")
    lines.append(f"0 {box} ylo yhi")
    lines.append(f"0 {box} zlo zhi")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    lines.append(f"1 {_fmt(network.config.mass_e)}")
    lines.append(f"2 {_fmt(network.config.mass_s)}")
    lines.append(f"3 {_fmt(network.config.mass_anchor)}")
    lines.append("")
    lines.append("Bond Coeffs # harmonic")
    lines.append("")
    lines.append("1 250 1")
    lines.append("2 250 0.5")
    lines.append("3 1 1")
    lines.append("4 1 1")
    lines.append("")
    lines.append("Atoms # hybrid")
    lines.append("")
    for atom in sorted(network.atoms, key=lambda item: item.atom_id):
        density = _density_for_atom(atom, network.config)
        lines.append(
            " ".join(
                [
                    str(atom.atom_id),
                    str(atom.atom_type),
                    _fmt(atom.x),
                    _fmt(atom.y),
                    _fmt(atom.z),
                    "1",
                    _fmt(density),
                    str(atom.molecule_id),
                    "0",
                    "0",
                    "0",
                ]
            )
        )
    lines.append("")
    lines.append("Velocities")
    lines.append("")
    for atom in sorted(network.atoms, key=lambda item: item.atom_id):
        lines.append(f"{atom.atom_id} 0 0 0 0 0 0")
    lines.append("")
    lines.append("Bonds")
    lines.append("")
    for bond in sorted(network.bonds, key=lambda item: item.bond_id):
        lines.append(f"{bond.bond_id} {bond.bond_type} {bond.atom1} {bond.atom2}")
    lines.append("")
    lines.append("Ellipsoids")
    lines.append("")
    for atom in sorted(network.atoms, key=lambda item: item.atom_id):
        sx, sy, sz = _shape_for_atom(atom, network.config)
        qw, qi, qj, qk = atom.quaternion
        lines.append(
            f"{atom.atom_id} {_fmt(sx)} {_fmt(sy)} {_fmt(sz)} "
            f"{_fmt(qw)} {_fmt(qi)} {_fmt(qj)} {_fmt(qk)}"
        )
    lines.append("")
    return "\n".join(lines)


def render_metadata_json(network: PolymerNetwork) -> str:
    payload = {
        "config": asdict(network.config),
        "counts": {
            "atoms": len(network.atoms),
            "bonds": len(network.bonds),
            "ellipsoids": network.count_atoms(1),
            "spheres": network.count_atoms(2),
            "anchors": network.count_atoms(3),
            "insertions": len(network.insertions),
            "horizontal_insertions": network.count_insertions("horizontal"),
            "vertical_insertions": network.count_insertions("vertical"),
            "x_insertions": network.count_insertions("x"),
            "y_insertions": network.count_insertions("y"),
            "z_insertions": network.count_insertions("z"),
        },
        "metadata": network.metadata,
        "insertions": [
            {
                "insertion_id": insertion.insertion_id,
                "segment_id": insertion.segment_id,
                "orientation": insertion.orientation,
                "start_index": insertion.start_index,
                "replaced_site_ids": sorted(insertion.replaced_site_ids),
                "molecule_id": insertion.molecule_id,
                "target_se_distance": insertion.target_se_distance,
                "actual_left_se_distance": insertion.actual_left_se_distance,
                "actual_right_se_distance": insertion.actual_right_se_distance,
            }
            for insertion in network.insertions
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_lammps_input(
    network: PolymerNetwork,
    data_filename: str,
    simulation: SimulationConfig | None = None,
) -> str:
    cfg = network.config
    sim = simulation or SimulationConfig()
    if _temperature_protocol(sim.temperature_protocol) == "cooldown":
        return _render_lammps_cooldown_input(network, data_filename, sim)

    data_reference = _lammps_path(data_filename)
    tstar_list = " ".join(sim.tstar_list)
    insertion_density_tag = sim.output_tag or _insertion_density_tag(cfg.insertion_density)
    tstar_map = "\n".join(
        f'if "${{Tstar_str}} == {value}" then "variable Tstar_tag string Tstar_{value}"'
        for value in tstar_list.split()
    )
    return f"""# Polymer network input generated from the sphere-chain parameter set.
# Geometry and topology are read from the generated data file; thermostat,
# temperature reporting, pair style, bond style, and output blocks follow the
# reference sphere-chain inputs.

dimension       3
boundary        p p p
units           lj
atom_style      hybrid ellipsoid bond

neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes
comm_style      tiled

variable        Nchain          equal 1
variable        Nell            equal {network.count_atoms(1)}
variable        Nsph            equal {network.count_atoms(2)}
variable        Nanchor         equal {network.count_atoms(3)}
variable        Nmol            equal {len(network.atoms)}
variable        Network_topology_mode string {cfg.topology_mode}
variable        Network_cells_x equal {cfg.cells_x}
variable        Network_cells_y equal {cfg.cells_y}
variable        Network_cells_z equal {cfg.cells_z}
variable        Horizontal_S_count equal {cfg.horizontal_s_count}
variable        Vertical_S_count equal {cfg.vertical_s_count}
variable        X_S_count equal {cfg.resolved_x_s_count}
variable        Y_S_count equal {cfg.resolved_y_s_count}
variable        Z_S_count equal {cfg.resolved_z_s_count}
variable        edge_gap        equal {_fmt(cfg.contact_gap)}

variable        HeatingT   equal {sim.heating_t}
variable        CoolingT   equal {sim.cooling_t}
variable        Tstar_list index {tstar_list}

variable        rho     equal {sim.rho}
variable        insertion_density_tag string {insertion_density_tag}
variable        TS      equal {sim.timestep}
variable        heatingTime      equal 10/${{TS}}
variable        coolingTime      equal 20/${{TS}}
variable        RunTime equal {sim.production_time_lj}/${{TS}}
variable        DT      equal ${{RunTime}}/{sim.dump_frames}
variable        DR      equal ${{DT}}*{sim.restart_interval_dump_frames}
variable        Tdamp   equal ${{TS}}*{sim.tdamp_factor}
variable        Tchain   equal {sim.tchain}

variable        k_bond_ss   equal {sim.bond_k_ss}
variable        k_bond_se   equal {sim.bond_k_se}
variable        r0_ss       equal {sim.r0_ss}
variable        r0_se       equal {sim.r0_se}
variable        ds_a        equal {sim.ds_a}
variable        dA          equal {sim.d_anchor}
variable        mA          equal {sim.m_anchor}
variable        r0_sa       equal {sim.r0_sa}
variable        r0_ae       equal {sim.r0_ae}

variable        dS_star equal 1
variable        dE_star equal 3
variable        d_shper equal 1
variable        dS      equal ${{dS_star}}
variable        dE      equal ${{dE_star}}
variable        a_semi0 equal ${{dS}}/2.0
variable        b_semi0 equal ${{dS}}/2.0
variable        c_semi0 equal ${{dE}}/2.0

variable        gb_gamma    equal {sim.gb_gamma}
variable        gb_upsilon  equal {sim.gb_upsilon}
variable        gb_mu       equal {sim.gb_mu}
variable        gb_rcutstar equal {sim.gb_rcutstar}
variable        epsilon0    equal {sim.epsilon0}
variable        sig0        equal {sim.sig0}
variable        eps_a       equal {sim.eps_a}
variable        eps_b       equal {sim.eps_b}
variable        eps_c       equal {sim.eps_c}
variable        gb_rcut     equal v_sig0*${{gb_rcutstar}}

variable        PI        equal 3.141592653589793
variable        Vp_ell    equal (4.0/3.0)*(v_PI*(v_a_semi0*v_b_semi0*v_c_semi0))
variable        Vp_sph    equal (4.0/3.0)*(v_PI*(0.5*0.5*0.5))
variable        Vp_anc    equal (4.0/3.0)*(v_PI*((v_dA/2.0)*(v_dA/2.0)*(v_dA/2.0)))
variable        Vtot_part equal (${{Nell}}*v_Vp_ell+${{Nsph}}*v_Vp_sph+${{Nanchor}}*v_Vp_anc)

bond_style      harmonic
read_data       {data_reference}

group           ellipsoid type 1
group           sphere    type 2
group           anchor    type 3
group           rigid_lc  type 1 3

variable        Nell_now equal count(ellipsoid)
variable        Nsph_now equal count(sphere)
variable        Nanc_now equal count(anchor)
print           "NETWORK COUNT CHECK: target Nell=${{Nell}}, target Nsph=${{Nsph}}, target Nanchor=${{Nanchor}}; actual Nell=${{Nell_now}}, actual Nsph=${{Nsph_now}}, actual Nanchor=${{Nanc_now}}"
print           "NETWORK GEOMETRY CHECK: S-S initial center distance = $(v_dS+v_edge_gap), target S-E center distance = $(0.5*v_dE+0.5*v_dS+v_edge_gap); preserve-grid metadata is in the generated JSON sidecar"

variable        dB      equal 1.0
variable        epsSS   equal 1.0
variable        sigSS   equal ${{dB}}
variable        epsSE   equal ${{epsilon0}}
variable        sigSE   equal ${{sig0}}
variable        rc_wca  equal 1.122462048309373*${{sigSS}}
variable        sigAA   equal ${{dA}}
variable        rcAA    equal 1.122462048309373*v_sigAA
variable        sigSA   equal 0.5*(${{sigSS}}+${{dA}})
variable        rcSA    equal 1.122462048309373*v_sigSA
variable        sigEA   equal 0.5*(${{sig0}}+${{dA}})
variable        rcEA    equal 1.122462048309373*v_sigEA

bond_coeff      1 ${{k_bond_ss}} ${{r0_ss}}
bond_coeff      2 ${{k_bond_ss}} ${{r0_sa}}
bond_coeff      3 1.0 1.0
bond_coeff      4 1.0 1.0
special_bonds   lj 0.0 1.0 1.0

pair_style      hybrid gayberne ${{gb_gamma}} ${{gb_upsilon}} ${{gb_mu}} ${{gb_rcut}} lj/cut ${{rc_wca}}
pair_modify     shift yes
pair_coeff      1 1 gayberne ${{epsilon0}} ${{sig0}} ${{eps_a}} ${{eps_b}} ${{eps_c}} ${{eps_a}} ${{eps_b}} ${{eps_c}} ${{gb_rcut}}
pair_coeff      2 2 lj/cut ${{epsSS}} ${{sigSS}} ${{rc_wca}}
pair_coeff      3 3 lj/cut 0.000001 ${{sigAA}} ${{rcAA}}
pair_coeff      2 3 lj/cut 0.000001 ${{sigSA}} ${{rcSA}}
pair_coeff      1 3 lj/cut 0.000001 ${{sigEA}} ${{rcEA}}
pair_coeff      1 2 gayberne ${{epsSE}} ${{sigSE}} ${{eps_a}} ${{eps_b}} ${{eps_c}} 1.0 1.0 1.0 ${{gb_rcut}}

compute         T_sph sphere temp
compute_modify  T_sph extra/dof 0
compute         KE_ell ellipsoid ke
compute         ER_ell ellipsoid erotate/asphere
variable        dof_sph   equal 3*count(sphere)
variable        dof_ell   equal 5*count(ellipsoid)
variable        dof_mix   equal v_dof_sph+v_dof_ell
variable        T_ell     equal 2.0*(c_KE_ell+c_ER_ell)/(v_dof_ell+1.0e-17)
variable        T_mix     equal (v_dof_sph*c_T_sph+v_dof_ell*v_T_ell)/v_dof_mix

compute         orient all property/atom quatw quati quatj quatk
compute         shape  all property/atom shapex shapey shapez

thermo          ${{DT}}
thermo_style    custom step temp v_T_mix c_T_sph v_T_ell v_dof_sph v_dof_ell v_dof_mix ebond pe ke etotal press vol v_Nell_now v_Nsph_now v_Nanc_now
thermo_modify   flush yes
timestep        ${{TS}}

label           loop_Tstar
variable        Tstar_str delete
variable        Tstar_str string ${{Tstar_list}}
print           "THERMOSTAT CHECK: both fixes use T*=${{Tstar_str}}; reported T_mix uses 5*NE + 3*NS DOF and ignores anchors"

variable        Tstar_tag delete
{tstar_map}

shell           mkdir -p ${{Tstar_tag}}
log             ${{Tstar_tag}}/log.${{Tstar_tag}}.lammps
print           "=== START TSTAR LOOP: T*=${{Tstar_str}}, folder=${{Tstar_tag}} ==="

variable        rho_str string {sim.rho_str}
variable        rho_star_inst equal v_Vtot_part/vol
variable        Lx_loop equal lx
variable        V_loop  equal vol
print           "RHO CHECK: fixed rho=${{rho_str}} actual lx=${{Lx_loop}} vol=${{V_loop}} inst rho(volume-fraction form)=${{rho_star_inst}}"

velocity        all create ${{Tstar_str}} {sim.velocity_seed} dist gaussian mom yes rot yes loop geom

variable        Nelli_now equal count(ellipsoid)
variable        Nsphi_now equal count(sphere)
variable        Nanci_now equal count(anchor)
print           "VELOCITY INIT CHECK: Nell(group)=${{Nelli_now}}, Nsph(group)=${{Nsphi_now}}, Nanchor(group)=${{Nanci_now}}; anchors and ellipsoids are rigidified as a-E-a units"
print           "OVITO OUTPUT CHECK: particle trajectory dump = HEAV.${{insertion_density_tag}}.*.dump ; static topology file = TOPOLOGY.${{insertion_density_tag}}.data"

dump            1 all custom ${{DT}} ${{Tstar_tag}}/HEAV.${{insertion_density_tag}}.*.dump id type x y z xu yu zu vx vy vz c_orient[1] c_orient[2] c_orient[3] c_orient[4] c_shape[1] c_shape[2] c_shape[3] mass
dump_modify     1 colname c_orient[1] quatw colname c_orient[2] quati colname c_orient[3] quatj colname c_orient[4] quatk
dump_modify     1 colname c_shape[1] shapex colname c_shape[2] shapey colname c_shape[3] shapez

compute         bond_topo all property/local btype batom1 batom2
write_data      ${{Tstar_tag}}/TOPOLOGY.${{insertion_density_tag}}.data
restart         ${{DR}} ${{Tstar_tag}}/Restart.GB.${{insertion_density_tag}}.*

compute         myRDF all rdf {sim.rdf_bins}
fix             RDFout all ave/time {sim.rdf_nevery} {sim.rdf_nrepeat} {sim.rdf_nfreq} c_myRDF[*] file ${{Tstar_tag}}/rdf.${{insertion_density_tag}}.dat mode vector

variable        TS_prod equal ${{TS}}
variable        kss_prod equal ${{k_bond_ss}}
variable        kse_prod equal ${{k_bond_se}}
timestep        {sim.pre_relax_timestep}
bond_coeff      1 {sim.pre_relax_bond_k} ${{r0_ss}}
bond_coeff      2 {sim.pre_relax_bond_k} ${{r0_sa}}
fix             temp_control_lc rigid_lc rigid/nvt/small molecule temp ${{Tstar_str}} ${{Tstar_str}} ${{Tdamp}}
fix             temp_control_sph sphere nvt temp ${{Tstar_str}} ${{Tstar_str}} ${{Tdamp}} tchain ${{Tchain}}
fix_modify      temp_control_sph temp T_sph
run             {sim.pre_relax_steps}

bond_coeff      1 ${{kss_prod}} ${{r0_ss}}
bond_coeff      2 ${{kss_prod}} ${{r0_sa}}
timestep        ${{TS_prod}}
run             ${{RunTime}}

unfix           temp_control_lc
unfix           temp_control_sph
unfix           RDFout
uncompute       myRDF
uncompute       bond_topo
undump          1

write_restart   ${{Tstar_tag}}/Final.${{insertion_density_tag}}.bin
log             none
print           "=== FINISH TSTAR LOOP: T*=${{Tstar_str}}, folder=${{Tstar_tag}} ==="
next            Tstar_list
jump            SELF loop_Tstar
"""


def _render_lammps_cooldown_input(
    network: PolymerNetwork,
    data_filename: str,
    sim: SimulationConfig,
) -> str:
    _validate_cooldown_cadence(sim)
    cfg = network.config
    data_reference = _lammps_path(data_filename)
    output_tag = sim.output_tag or _insertion_density_tag(cfg.insertion_density)
    tstar_values = _cooldown_tstar_values(sim.tstar_list)
    has_lc = network.count_atoms(1) > 0
    compute_ell_block = (
        """compute         KE_ell ellipsoid ke
compute         ER_ell ellipsoid erotate/asphere
variable        dof_ell   equal 5*count(ellipsoid)
variable        T_ell     equal 2.0*(c_KE_ell+c_ER_ell)/(v_dof_ell+1.0e-17)"""
        if has_lc
        else """variable        dof_ell   equal 0
variable        T_ell     equal 0.0"""
    )
    prehold_lc_fix = (
        f'if "${{PreHoldSteps}} > 0" then "fix temp_control_lc rigid_lc rigid/nvt/small molecule temp {tstar_values[0]} {tstar_values[0]} ${{Tdamp}}"\n'
        if has_lc
        else ""
    )
    prehold_lc_unfix = (
        'if "${PreHoldSteps} > 0" then "unfix temp_control_lc"\n'
        if has_lc
        else ""
    )
    stage_blocks = [
        _cooldown_phase_block(
            tstar_values[0],
            tstar_values[0],
            sim,
            output_tag,
            has_lc,
            phase="ramp",
            run_steps_var="${RampSteps}",
            dump_every_var="${DTRamp}",
            label="high-temperature ramp-equilibration",
        ),
        _cooldown_phase_block(
            tstar_values[0],
            tstar_values[0],
            sim,
            output_tag,
            has_lc,
            phase="relax",
            run_steps_var="${RelaxSteps}",
            dump_every_var="${DTRelax}",
            label="high-temperature relax",
        ),
        _cooldown_phase_block(
            tstar_values[0],
            tstar_values[0],
            sim,
            output_tag,
            has_lc,
            phase="sample",
            run_steps_var="${SampleSteps}",
            dump_every_var="${DTSample}",
            label="high-temperature sample",
            volume_state=True,
            write_outputs=True,
        ),
    ]
    for previous_tstar, tstar in zip(tstar_values, tstar_values[1:]):
        stage_blocks.append(
            _cooldown_phase_block(
                previous_tstar,
                tstar,
                sim,
                output_tag,
                has_lc,
                phase="ramp",
                run_steps_var="${RampSteps}",
                dump_every_var="${DTRamp}",
                label="cooling ramp",
            )
        )
        stage_blocks.append(
            _cooldown_phase_block(
                tstar,
                tstar,
                sim,
                output_tag,
                has_lc,
                phase="relax",
                run_steps_var="${RelaxSteps}",
                dump_every_var="${DTRelax}",
                label="post-ramp relax",
            )
        )
        stage_blocks.append(
            _cooldown_phase_block(
                tstar,
                tstar,
                sim,
                output_tag,
                has_lc,
                phase="sample",
                run_steps_var="${SampleSteps}",
                dump_every_var="${DTSample}",
                label="production sample",
                volume_state=True,
                write_outputs=True,
            )
        )
    stages = "\n".join(stage_blocks)
    return f"""# Polymer network cooldown input generated from the a-E-a network generator.
# COOLDOWN PROTOCOL: high-temperature ramp/relax/sample followed by staged
# ramp/relax/sample cooling. The legacy temperature_sweep protocol remains available
# through render_lammps_input(..., SimulationConfig.temperature_protocol).

dimension       3
boundary        p p p
units           lj
atom_style      hybrid ellipsoid bond

neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes
comm_style      tiled

variable        Nchain          equal 1
variable        Nell            equal {network.count_atoms(1)}
variable        Nsph            equal {network.count_atoms(2)}
variable        Nanchor         equal {network.count_atoms(3)}
variable        Network_topology_mode string {cfg.topology_mode}
variable        Network_cells_x equal {cfg.cells_x}
variable        Network_cells_y equal {cfg.cells_y}
variable        Network_cells_z equal {cfg.cells_z}
variable        X_S_count equal {cfg.resolved_x_s_count}
variable        Y_S_count equal {cfg.resolved_y_s_count}
variable        Z_S_count equal {cfg.resolved_z_s_count}
variable        edge_gap        equal {_fmt(cfg.contact_gap)}

variable        Tstart   equal {tstar_values[0]}
variable        Tstop    equal {tstar_values[-1]}
variable        TS       equal {sim.timestep}
variable        PreHoldSteps equal {sim.cooldown_pre_hold_time_lj}/${{TS}}
variable        RampSteps    equal {sim.cooldown_ramp_time_lj}/${{TS}}
variable        RelaxSteps   equal {sim.cooldown_relax_time_lj}/${{TS}}
variable        SampleSteps  equal {sim.cooldown_sample_time_lj}/${{TS}}
variable        DTRamp       equal ${{RampSteps}}/{sim.ramp_dump_frames}
variable        DTRelax      equal ${{RelaxSteps}}/{sim.relax_dump_frames}
variable        DTSample     equal ${{SampleSteps}}/{sim.sample_dump_frames}
variable        DR           equal ${{DTSample}}*{sim.restart_interval_dump_frames}
variable        Tdamp    equal ${{TS}}*{sim.tdamp_factor}
variable        Tchain   equal {sim.tchain}
variable        output_tag string {output_tag}

variable        k_bond_ss   equal {sim.bond_k_ss}
variable        k_bond_se   equal {sim.bond_k_se}
variable        r0_ss       equal {sim.r0_ss}
variable        r0_se       equal {sim.r0_se}
variable        ds_a        equal {sim.ds_a}
variable        dA          equal {sim.d_anchor}
variable        mA          equal {sim.m_anchor}
variable        r0_sa       equal {sim.r0_sa}
variable        r0_ae       equal {sim.r0_ae}

variable        dB      equal 1.0
variable        epsSS   equal 1.0
variable        sigSS   equal ${{dB}}
variable        epsSE   equal {sim.epsilon0}
variable        sigSE   equal {sim.sig0}
variable        rc_wca  equal 1.122462048309373*${{sigSS}}
variable        sigAA   equal ${{dA}}
variable        rcAA    equal 1.122462048309373*v_sigAA
variable        sigSA   equal 0.5*(${{sigSS}}+${{dA}})
variable        rcSA    equal 1.122462048309373*v_sigSA
variable        sigEA   equal 0.5*({sim.sig0}+${{dA}})
variable        rcEA    equal 1.122462048309373*v_sigEA
variable        gb_gamma    equal {sim.gb_gamma}
variable        gb_upsilon  equal {sim.gb_upsilon}
variable        gb_mu       equal {sim.gb_mu}
variable        gb_rcutstar equal {sim.gb_rcutstar}
variable        epsilon0    equal {sim.epsilon0}
variable        sig0        equal {sim.sig0}
variable        eps_a       equal {sim.eps_a}
variable        eps_b       equal {sim.eps_b}
variable        eps_c       equal {sim.eps_c}
variable        gb_rcut     equal v_sig0*${{gb_rcutstar}}

bond_style      harmonic
read_data       {data_reference}

group           ellipsoid type 1
group           sphere    type 2
group           anchor    type 3
group           rigid_lc  type 1 3

bond_coeff      1 ${{k_bond_ss}} ${{r0_ss}}
bond_coeff      2 ${{k_bond_ss}} ${{r0_sa}}
bond_coeff      3 1.0 1.0
bond_coeff      4 1.0 1.0
special_bonds   lj 0.0 1.0 1.0

pair_style      hybrid gayberne ${{gb_gamma}} ${{gb_upsilon}} ${{gb_mu}} ${{gb_rcut}} lj/cut ${{rc_wca}}
pair_modify     shift yes
pair_coeff      1 1 gayberne ${{epsilon0}} ${{sig0}} ${{eps_a}} ${{eps_b}} ${{eps_c}} ${{eps_a}} ${{eps_b}} ${{eps_c}} ${{gb_rcut}}
pair_coeff      2 2 lj/cut ${{epsSS}} ${{sigSS}} ${{rc_wca}}
pair_coeff      3 3 lj/cut 0.000001 ${{sigAA}} ${{rcAA}}
pair_coeff      2 3 lj/cut 0.000001 ${{sigSA}} ${{rcSA}}
pair_coeff      1 3 lj/cut 0.000001 ${{sigEA}} ${{rcEA}}
pair_coeff      1 2 gayberne ${{epsSE}} ${{sigSE}} ${{eps_a}} ${{eps_b}} ${{eps_c}} 1.0 1.0 1.0 ${{gb_rcut}}

compute         T_sph sphere temp
compute_modify  T_sph extra/dof 0
{compute_ell_block}
variable        dof_sph   equal 3*count(sphere)
variable        dof_mix   equal v_dof_sph+v_dof_ell
variable        T_mix     equal (v_dof_sph*c_T_sph+v_dof_ell*v_T_ell)/(v_dof_mix+1.0e-17)

compute         orient all property/atom quatw quati quatj quatk
compute         shape  all property/atom shapex shapey shapez
compute         pos all property/atom x y z
compute         xmin all reduce min c_pos[1]
compute         xmax all reduce max c_pos[1]
compute         ymin all reduce min c_pos[2]
compute         ymax all reduce max c_pos[2]
compute         zmin all reduce min c_pos[3]
compute         zmax all reduce max c_pos[3]

variable        step_now equal step
variable        time_now equal time
variable        temp_now equal temp
variable        press_now equal press
variable        lx_now equal lx
variable        ly_now equal ly
variable        lz_now equal lz
variable        vol_now equal vol

thermo          ${{DTSample}}
thermo_style    custom step temp v_T_mix c_T_sph v_T_ell v_dof_sph v_dof_ell v_dof_mix ebond pe ke etotal press vol
thermo_modify   flush yes lost warn
timestep        ${{TS}}

velocity        all create {tstar_values[0]} {sim.velocity_seed} dist gaussian mom yes rot yes loop geom
{prehold_lc_fix}if "${{PreHoldSteps}} > 0" then "fix temp_control_sph sphere nvt temp {tstar_values[0]} {tstar_values[0]} ${{Tdamp}} tchain ${{Tchain}}"
if "${{PreHoldSteps}} > 0" then "fix_modify temp_control_sph temp T_sph"
if "${{PreHoldSteps}} > 0" then "run ${{PreHoldSteps}}"
{prehold_lc_unfix}if "${{PreHoldSteps}} > 0" then "unfix temp_control_sph"

{stages}
"""


def _cooldown_phase_block(
    previous_tstar: str,
    target_tstar: str,
    sim: SimulationConfig,
    output_tag: str,
    has_lc: bool,
    *,
    phase: str,
    run_steps_var: str,
    dump_every_var: str,
    label: str,
    volume_state: bool = False,
    write_outputs: bool = False,
) -> str:
    tag = f"Tstar_{target_tstar}"
    lc_fix = (
        f"fix             temp_control_lc rigid_lc rigid/nvt/small molecule temp {previous_tstar} {target_tstar} ${{Tdamp}}\n"
        if has_lc
        else ""
    )
    lc_unfix = "unfix           temp_control_lc\n" if has_lc else ""
    volume_state_block = (
        f'fix             volume_state all print {sim.volume_sample_every} "${{step_now}} ${{time_now}} ${{temp_now}} ${{press_now}} ${{lx_now}} ${{ly_now}} ${{lz_now}} ${{vol_now}} $(c_xmin) $(c_xmax) $(c_ymin) $(c_ymax) $(c_zmin) $(c_zmax)" file {tag}/volume_state.{output_tag}.{phase}.dat screen no title "# step time temp press lx ly lz box_vol xmin xmax ymin ymax zmin zmax"\n'
        if volume_state
        else ""
    )
    volume_state_unfix = "unfix           volume_state\n" if volume_state else ""
    output_block = (
        f"""write_data      {tag}/TOPOLOGY.{output_tag}.data
write_restart   {tag}/Final.cooldown.{output_tag}.bin
"""
        if write_outputs
        else ""
    )
    return f"""shell           mkdir -p {tag}
log             {tag}/log.{tag}.{phase}.lammps
print           "COOLDOWN {label}: T*={previous_tstar} -> {target_tstar}, folder={tag}, phase={phase}"
thermo          {dump_every_var}
dump            traj all custom {dump_every_var} {tag}/HEAV.{output_tag}.{phase}.*.dump id type x y z xu yu zu vx vy vz c_orient[1] c_orient[2] c_orient[3] c_orient[4] c_shape[1] c_shape[2] c_shape[3] mass
dump_modify     traj colname c_orient[1] quatw colname c_orient[2] quati colname c_orient[3] quatj colname c_orient[4] quatk
dump_modify     traj colname c_shape[1] shapex colname c_shape[2] shapey colname c_shape[3] shapez
restart         ${{DR}} {tag}/Restart.cooldown.{output_tag}.*
{volume_state_block}{lc_fix}fix             temp_control_sph sphere nvt temp {previous_tstar} {target_tstar} ${{Tdamp}} tchain ${{Tchain}}
fix_modify      temp_control_sph temp T_sph
run             {run_steps_var}
{lc_unfix}unfix           temp_control_sph
{volume_state_unfix}
undump          traj
{output_block}thermo          ${{DTSample}}
log             none
"""


def _shape_for_atom(atom: Atom, config: NetworkConfig) -> tuple[float, float, float]:
    if atom.atom_type == 1:
        return (config.e_short_diameter, config.e_short_diameter, config.e_long_diameter)
    if atom.atom_type == 2:
        return (config.s_diameter, config.s_diameter, config.s_diameter)
    return (config.anchor_diameter, config.anchor_diameter, config.anchor_diameter)


def _density_for_atom(atom: Atom, config: NetworkConfig) -> float:
    shape = _shape_for_atom(atom, config)
    volume = (4.0 / 3.0) * 3.141592653589793 * (shape[0] / 2.0) * (shape[1] / 2.0) * (shape[2] / 2.0)
    if atom.atom_type == 1:
        return config.mass_e / volume
    if atom.atom_type == 2:
        return config.mass_s / volume
    return config.mass_anchor / volume


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.15g}"


def _insertion_density_tag(value: float) -> str:
    scaled = int(float(value) * 100.0 + 0.5)
    return f"rho{scaled:03d}"


def _temperature_protocol(value: str) -> str:
    if value == "independent":
        return "temperature_sweep"
    if value not in ("temperature_sweep", "cooldown"):
        raise ValueError("temperature_protocol must be 'temperature_sweep' or 'cooldown'")
    return value


def _cooldown_tstar_values(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        raise ValueError("cooldown tstar_list must contain at least one value")
    numeric = [float(value) for value in values]
    for left, right in zip(numeric, numeric[1:]):
        if left < right:
            raise ValueError("cooldown tstar_list must be in descending temperature order")
    return tuple(str(value) for value in values)


def _validate_cooldown_cadence(sim: SimulationConfig) -> None:
    timestep = Decimal(str(sim.timestep))
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    if sim.ramp_dump_frames <= 0:
        raise ValueError("ramp_dump_frames must be positive")
    if sim.relax_dump_frames <= 0:
        raise ValueError("relax_dump_frames must be positive")
    if sim.sample_dump_frames <= 0:
        raise ValueError("sample_dump_frames must be positive")
    if sim.restart_interval_dump_frames <= 0:
        raise ValueError("restart_interval_dump_frames must be positive")
    if sim.volume_sample_every <= 0:
        raise ValueError("volume_sample_every must be positive")

    pre_hold_steps = _integer_steps(sim.cooldown_pre_hold_time_lj, timestep, "cooldown_pre_hold_time_lj")
    ramp_steps = _integer_steps(sim.cooldown_ramp_time_lj, timestep, "cooldown_ramp_time_lj")
    relax_steps = _integer_steps(sim.cooldown_relax_time_lj, timestep, "cooldown_relax_time_lj")
    sample_steps = _integer_steps(sim.cooldown_sample_time_lj, timestep, "cooldown_sample_time_lj")
    if pre_hold_steps < 0 or ramp_steps <= 0 or relax_steps <= 0 or sample_steps <= 0:
        raise ValueError("cooldown pre-hold must be non-negative and ramp/relax/sample times must be positive")
    if ramp_steps % sim.ramp_dump_frames != 0:
        raise ValueError("RampSteps/ramp_dump_frames must be an integer")
    if relax_steps % sim.relax_dump_frames != 0:
        raise ValueError("RelaxSteps/relax_dump_frames must be an integer")
    if sample_steps % sim.sample_dump_frames != 0:
        raise ValueError("SampleSteps/sample_dump_frames must be an integer")


def _integer_steps(time_lj: str, timestep: Decimal, field_name: str) -> int:
    steps = Decimal(str(time_lj)) / timestep
    integral = steps.to_integral_value()
    if steps != integral:
        raise ValueError(f"{field_name}/timestep must be an integer number of steps")
    return int(integral)


def _lammps_path(path: str) -> str:
    if any(character.isspace() for character in path):
        escaped = path.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return path
