

particles = ['H_calc', 'a1' ,'a2'] 
targets = ['mass', 'logMass', '1OverMass', 'massOverPT', 'logMassOverPT' ,'ptOverMass']
eqns = ['fj_gen_H_aa_bbbb_mass_{particle}',
        'np.log(fj_gen_H_aa_bbbb_mass_{particle})',
        '1/fj_gen_H_aa_bbbb_mass_{particle}',
        'fj_gen_H_aa_bbbb_mass_{particle}/fj_pt',
        'np.log(fj_gen_H_aa_bbbb_mass_{particle}/fj_pt)',
        'fj_pt/fj_gen_H_aa_bbbb_mass_{particle}'
    ] 

variables = [f'fj_gen_H_aa_bbbb_{target}_{particle}' for target in targets for particle in particles]
eqns = [eqn.format(particle=particle) for eqn in eqns for particle in particles ]
filenames = [f'{particle}_{target}_regr.yaml' for target in targets for particle in particles]

txt = '''
selection:
   (label_H_aa_bbbb == 1) & (fj_pt > 170) & ((event_no % 2) == 0) 
   ## selection for not Wide H. for wide H we have lower mass H, so we can have lower pt that still H 
   ## (label_H_aa_bbbb == 1) & (fj_pt > 200) & (fj_mass > 50) & (pfMassDecorrelatedParticleNetDiscriminatorsJetTags_XbbvsQCD > 0.1) & ((event_no % 2) == 0)
   

test_time_selection:
   (fj_mass > 0) & ((event_no % 2) == 1)

new_variables:
   pfcand_mask: awkward.JaggedArray.ones_like(pfcand_etarel)
   sv_mask: awkward.JaggedArray.ones_like(sv_etarel)
   {variable} : {eqn}

preprocess:
  method: manual
  data_fraction: 

inputs:
   pf_points:
      length: 100
      vars: 
         - pfcand_etarel
         - pfcand_phirel
   pf_features:
      length: 100
      vars: 
         - [pfcand_pt_log_nopuppi, 1, 0.5]
         - [pfcand_e_log_nopuppi, 1.3, 0.5]
         - pfcand_etarel
         - pfcand_phirel
         - [pfcand_abseta, 0.6, 1.6]
         - pfcand_charge
         - pfcand_isEl
         - pfcand_isMu
         - pfcand_isGamma
         - pfcand_isChargedHad
         - pfcand_isNeutralHad
         - [pfcand_VTX_ass, 4, 0.3]
         - pfcand_lostInnerHits
         - [pfcand_normchi2, 5, 0.2]
         - [pfcand_quality, 0, 0.2]
         - [pfcand_dz, 0, 180]
         - [pfcand_dzsig, 0, 0.9]
         - [pfcand_dxy, 0.0, 300]
         - [pfcand_dxysig, 0, 1.0]
         - [pfcand_btagEtaRel, 1.5, 0.5]
         - [pfcand_btagPtRatio, 0, 1]
         - [pfcand_btagPParRatio, 0, 1]
         - [pfcand_btagSip3dVal, 0, 100]
         - [pfcand_btagSip3dSig, 0, 0.5]
         - [pfcand_btagJetDistVal, 0, 40]
   pf_mask:
      length: 100
      vars: 
         - pfcand_mask
   sv_points:
      length: 10
      vars:
         - sv_etarel 
         - sv_phirel
   sv_features:
      length: 10
      vars:
         - [sv_pt_log, 4, 0.6]
         - [sv_mass, 1.2, 0.3]
         - sv_etarel
         - sv_phirel
         - [sv_abseta, 0.5, 1.6]
         - [sv_ntracks, 3, 1]
         - [sv_normchi2, 0.8, 0.6]
         - [sv_dxy, 0.4, 0.25]
         - [sv_dxysig, 7, 0.02]
         - [sv_d3d, 0.5, 0.2]
         - [sv_d3dsig, 7, 0.02]
   sv_mask:
      length: 10
      vars:
         - sv_mask

labels:
   type: custom
   value: 
      target_mass: {variable}

observers:
   - event_no
   - jet_no
   - npv
   - label_H_aa_bbbb
   - label_H_aa_other
   - label_QCD_BGen
   - label_QCD_bEnr
   - sample_isQCD
   - sample_min_LHE_HT
   - fj_pt
   - fj_eta
   - fj_mass
   - fj_sdmass
   - fj_corrsdmass
   - fj_sdmass_fromsubjets
   - fj_gen_pt
   - fj_genjet_pt
   - fj_genjet_mass
   - fj_genjet_sdmass
   - fj_gen_H_aa_bbbb_mass_a1
   - fj_gen_H_aa_bbbb_mass_a2
   - fj_gen_H_aa_bbbb_mass_H
   - fj_gen_H_aa_bbbb_mass_H_calc
   - fj_gen_H_aa_bbbb_dR_max_b
   - fj_gen_H_aa_bbbb_pt_min_b
   - pfParticleNetMassRegressionJetTags_mass
   - pfParticleNetDiscriminatorsJetTags_HbbvsQCD
   - pfParticleNetDiscriminatorsJetTags_H4qvsQCD
   - pfMassDecorrelatedParticleNetDiscriminatorsJetTags_XbbvsQCD
'''



for variable, eqn, fn in zip(variables, eqns, filenames):
    print(txt.format(variable=variable, eqn=eqn))
    with open(fn, 'w+') as f:
        f.write(txt.format(variable=variable, eqn=eqn))
