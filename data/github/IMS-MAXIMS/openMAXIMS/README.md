# https://github.com/IMS-MAXIMS/openMAXIMS

```console
Source Library/database/Reference-Data-scripts-rel147-openMAXIMS.sql:insert into applookup (id, name, description, active, hierarchical, systemtype) values( 1231059, 'DelayReasonGPUrgentReferral', 'DelayReasonGPUrgentReferral', 1, 0, 0)
Source Library/database/Reference-Data-scripts-rel147-openMAXIMS.sql:insert into core_appcontextvari (id, vstp, rie, variablena, variableke, iscollecti, lkp_variablety, lkp_valuetype, valueclass) values( 1583, 0, 0, 'Emergency.TrackingPublicArea', '_cvp_Emergency.TrackingPublicArea', 0, -1178, -1183, 'Bool')
Source Library/openmaxims_workspace/WebApp/MAXIMS_RefDataORA.sql:insert into applookup (id, name, description, active, hierarchical, systemtype) values( 1231059, 'DelayReasonGPUrgentReferral', 'DelayReasonGPUrgentReferral', 1, 0, 0)
Source Library/openmaxims_workspace/WebApp/MAXIMS_RefDataORA.sql:insert into core_appcontextvari (id, vstp, rie, variablena, variableke, iscollecti, lkp_variablety, lkp_valuetype, valueclass) values( 1583, 0, 0, 'Emergency.TrackingPublicArea', '_cvp_Emergency.TrackingPublicArea', 0, -1178, -1183, 'Bool ')
Source Library/openmaxims_workspace/WebApp/InsertLookupBoMapping.sql:insert into applookup_type_col_map(table_name, col_name, type_id, bo_name, bo_field) values('shcl_referralsrecor', 'lkp_delayreaso', 1231059, 'ReferralsRecording', 'delayReasonGPUrgentReferral')
Source Library/openmaxims_workspace/WebApp/dictionary.dic:barracuda
Source Library/openmaxims_workspace/WebApp/dictionary.dic:barracuda's
Source Library/openmaxims_workspace/WebApp/dictionary.dic:barracudas
Source Library/openmaxims_workspace/WebApp/MAXIMS_RefDataMSQ05.sql:insert into applookup (id, name, description, active, hierarchical, systemtype) values( 1231059, 'DelayReasonGPUrgentReferral', 'DelayReasonGPUrgentReferral', 1, 0, 0)
Source Library/openmaxims_workspace/WebApp/MAXIMS_RefDataMSQ05.sql:insert into core_appcontextvari (id, vstp, rie, variablena, variableke, iscollecti, lkp_variablety, lkp_valuetype, valueclass) values( 1583, 0, 0, 'Emergency.TrackingPublicArea', '_cvp_Emergency.TrackingPublicArea', 0, -1178, -1183, 'Bool')
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:		if(this.delayreasongpurgentreferral == null)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:			clone.delayreasongpurgentreferral = null;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:			clone.delayreasongpurgentreferral = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)this.delayreasongpurgentreferral.clone();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:	public ims.vo.LookupInstanceBean getDelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		return this.delayreasongpurgentreferral;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:	public void setDelayReasonGPUrgentReferral(ims.vo.LookupInstanceBean value)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		this.delayreasongpurgentreferral = value;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:	private ims.vo.LookupInstanceBean delayreasongpurgentreferral;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:	public ims.vo.LookupInstanceBean getDelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		return this.delayreasongpurgentreferral;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:	public void setDelayReasonGPUrgentReferral(ims.vo.LookupInstanceBean value)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		this.delayreasongpurgentreferral = value;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:	private ims.vo.LookupInstanceBean delayreasongpurgentreferral;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		if(fieldName.equals("DELAYREASONGPURGENTREFERRAL"))
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:			return getDelayReasonGPUrgentReferral();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	public boolean getDelayReasonGPUrgentReferralIsNotNull()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		return this.delayreasongpurgentreferral != null;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	public ims.clinical.vo.lookups.DelayReasonGPUrgentReferral getDelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		return this.delayreasongpurgentreferral;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	public void setDelayReasonGPUrgentReferral(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		this.delayreasongpurgentreferral = value;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		if(this.delayreasongpurgentreferral == null)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:			clone.delayreasongpurgentreferral = null;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:			clone.delayreasongpurgentreferral = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)this.delayreasongpurgentreferral.clone();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		if(this.delayreasongpurgentreferral != null)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	protected ims.clinical.vo.lookups.DelayReasonGPUrgentReferral delayreasongpurgentreferral;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:public class DelayReasonGPUrgentReferralCollection extends LookupInstanceCollection implements ims.vo.ImsCloneable, TreeModel
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public void add(DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public int indexOf(DelayReasonGPUrgentReferral instance)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public boolean contains(DelayReasonGPUrgentReferral instance)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public DelayReasonGPUrgentReferral get(int index)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		return (DelayReasonGPUrgentReferral)super.getIndex(index);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public void remove(DelayReasonGPUrgentReferral instance)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferralCollection newCol = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferral item;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:			newCol.add(new DelayReasonGPUrgentReferral(item.getID(), item.getText(), item.isActive(), item.getParent(), item.getImage(), item.getColor(), item.getOrder()));
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:					item.setParent((DelayReasonGPUrgentReferral)item.getParent().clone());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public DelayReasonGPUrgentReferral getInstance(int instanceId)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		return (DelayReasonGPUrgentReferral)super.getInstanceById(instanceId);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public DelayReasonGPUrgentReferral[] toArray()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferral[] arr = new DelayReasonGPUrgentReferral[this.size()];
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public static DelayReasonGPUrgentReferralCollection buildFromBeanCollection(java.util.Collection beans)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferralCollection coll = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:			coll.add(DelayReasonGPUrgentReferral.buildLookup((ims.vo.LookupInstanceBean)iter.next()));
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public static DelayReasonGPUrgentReferralCollection buildFromBeanCollection(ims.vo.LookupInstanceBean[] beans)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferralCollection coll = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:			coll.add(DelayReasonGPUrgentReferral.buildLookup(beans[x]));
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:	public static DelayReasonGPUrgentReferralCollection getDelayReasonGPUrgentReferral(LookupService lookupService) {
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:		DelayReasonGPUrgentReferralCollection collection =
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:			(DelayReasonGPUrgentReferralCollection) lookupService.getLookupCollection(DelayReasonGPUrgentReferral.TYPE_ID, 
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:				DelayReasonGPUrgentReferralCollection.class, DelayReasonGPUrgentReferral.class);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:	public static DelayReasonGPUrgentReferral getDelayReasonGPUrgentReferralInstance(LookupService lookupService, int id) 
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:		return (DelayReasonGPUrgentReferral)lookupService.getLookupInstance(DelayReasonGPUrgentReferral.class, DelayReasonGPUrgentReferral.TYPE_ID, id);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:public class DelayReasonGPUrgentReferral extends ims.vo.LookupInstVo implements TreeNode
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active, DelayReasonGPUrgentReferral parent, Image image)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active, DelayReasonGPUrgentReferral parent, Image image, Color color)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active, DelayReasonGPUrgentReferral parent, Image image, Color color, int order)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral buildLookup(ims.vo.LookupInstanceBean bean)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return new DelayReasonGPUrgentReferral(bean.getId(), bean.getText(), bean.isActive());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return (DelayReasonGPUrgentReferral)super.getParentInstance();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral getParent()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return (DelayReasonGPUrgentReferral)super.getParentInstance();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public void setParent(DelayReasonGPUrgentReferral parent)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		DelayReasonGPUrgentReferral[] typedChildren = new DelayReasonGPUrgentReferral[children.size()];
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:			typedChildren[i] = (DelayReasonGPUrgentReferral)children.get(i);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		if (child instanceof DelayReasonGPUrgentReferral)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:			super.addChild((DelayReasonGPUrgentReferral)child);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		if (child instanceof DelayReasonGPUrgentReferral)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:			super.removeChild((DelayReasonGPUrgentReferral)child);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		DelayReasonGPUrgentReferralCollection result = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral[] getNegativeInstances()
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return new DelayReasonGPUrgentReferral[] {};
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral getNegativeInstance(String name)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral getNegativeInstance(Integer id)
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		valueObjectDest.setDelayReasonGPUrgentReferral(valueObjectSrc.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:				// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		ims.domain.lookups.LookupInstance instance29 = domainObject.getDelayReasonGPUrgentReferral();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral voLookup29 = new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(instance29.getId(),instance29.getText(), instance29.isActive(), null, img, color);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral parentVoLookup29 = voLookup29;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:								parentVoLookup29.setParent(new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(parent29.getId(),parent29.getText(), parent29.isActive(), null, img, color));
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:			valueObject.setDelayReasonGPUrgentReferral(voLookup29);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		if ( null != valueObject.getDelayReasonGPUrgentReferral() ) 
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:				domainFactory.getLookupInstance(valueObject.getDelayReasonGPUrgentReferral().getID());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		domainObject.setDelayReasonGPUrgentReferral(value29);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		valueObjectDest.setDelayReasonGPUrgentReferral(valueObjectSrc.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:				// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		ims.domain.lookups.LookupInstance instance25 = domainObject.getDelayReasonGPUrgentReferral();
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral voLookup25 = new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(instance25.getId(),instance25.getText(), instance25.isActive(), null, img, color);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral parentVoLookup25 = voLookup25;
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:								parentVoLookup25.setParent(new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(parent25.getId(),parent25.getText(), parent25.isActive(), null, img, color));
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:			valueObject.setDelayReasonGPUrgentReferral(voLookup25);
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		if ( null != valueObject.getDelayReasonGPUrgentReferral() ) 
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:				domainFactory.getLookupInstance(valueObject.getDelayReasonGPUrgentReferral().getID());
Source Library/openmaxims_workspace/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		domainObject.setDelayReasonGPUrgentReferral(value25);
Source Library/openmaxims_workspace/ValueObjects/src/ims/vo/lookups/ClassHelper.java:				return ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.class;
Source Library/openmaxims_workspace/ValueObjects/src/ims/vo/lookups/ClassHelper.java:				return ims.clinical.vo.lookups.DelayReasonGPUrgentReferralCollection.class;
Source Library/openmaxims_workspace/ICP/src/ims/icp/forms/patienticpactiondetails/Handlers.java:	abstract protected void onCcLinkedActionDetailsValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/ICP/src/ims/icp/forms/patienticpactiondetails/Handlers.java:				onCcLinkedActionDetailsValueChanged();
Source Library/openmaxims_workspace/ICP/src/ims/icp/forms/patienticpactiondetails/Logic.java:	protected void onCcLinkedActionDetailsValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/Logic.java:		form.cmbDelayReasonFirstSeen().setValue(rrVo.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/Logic.java:		rrVo.setDelayReasonGPUrgentReferral(form.cmbDelayReasonFirstSeen().getValue());
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text)
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text, ims.framework.utils.Image image)
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text, ims.framework.utils.Color textColor)
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text, ims.framework.utils.Image image, ims.framework.utils.Color textColor)
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public boolean removeRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public ims.clinical.vo.lookups.DelayReasonGPUrgentReferral getValue()
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:			return (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)super.control.getValue();
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void setValue(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:			fields[162] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedPreceedingReferralVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:			fields[184] = new ims.framework.ReportField(this.context, prefix + "_lv_Oncology.CancerreferralDetails.__internal_x_context__SelectedReferral_" + componentIdentifier, "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:					ims.clinical.vo.lookups.DelayReasonGPUrgentReferral existingInstance = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)listOfValues.get(x);
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		else if(value instanceof ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral instance = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)value;
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:					ims.clinical.vo.lookups.DelayReasonGPUrgentReferral existingInstance = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)listOfValues.get(x);
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		ims.clinical.vo.lookups.DelayReasonGPUrgentReferralCollection lookupCollection = ims.clinical.vo.lookups.LookupHelper.getDelayReasonGPUrgentReferral(this.domain.getLookupService());
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		ims.clinical.vo.lookups.DelayReasonGPUrgentReferral instance = ims.clinical.vo.lookups.LookupHelper.getDelayReasonGPUrgentReferralInstance(this.domain.getLookupService(), id);
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		this.form.cmbDelayReasonFirstSeen().setValue((ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)domain.getLookupService().getDefaultInstance(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.class, engine.getFormName().getID(), ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.TYPE_ID));
Source Library/openmaxims_workspace/Oncology/src/ims/oncology/domain/impl/WeekWaitingTimesImpl.java:				domWT.setDelayReasonRefFirstSeen(getExternalLookup(domRef.getDelayReasonGPUrgentReferral()));
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/referralsrecording/GenForm.java:			fields[162] = new ims.framework.ReportField(this.context, prefix + "_lv_Clinical.ReferralsRecording.__internal_x_context__EditedRecord_" + componentIdentifier, "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/cliniclistactionsummary/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/cliniclistactionsummary/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/cliniclistactionsummary/Logic.java:	protected void onCcListOwnerValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patientinternalreferralslist/Handlers.java:	abstract protected void onCcListGridValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patientinternalreferralslist/Handlers.java:				onCcListGridValueChanged();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patientinternalreferralslist/Logic.java:	protected void onCcListGridValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/surgicalauditpreoperationchecks/GenForm.java:	public TextBox txtProcMandatory()
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/demoreferrallist/GenForm.java:			fields[162] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedReferralRecordingVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patienticplist/Logic.java:				openCloseICP();//WDEV-12965
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patienticplist/Logic.java:				openCloseICP();//WDEV-12965
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patienticplist/Logic.java:	private void openCloseICP()//WDEV-12965
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/acutetheatrelist/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/acutetheatrelist/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/acutetheatrelist/Logic.java:	protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patient_summary/GenForm.java:		public void disableAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patient_summary/GenForm.java:		public void hideAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/outpatientcliniclist/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/outpatientcliniclist/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/outpatientcliniclist/Logic.java:	protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:					openClinicalNotes();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:					openClinicalNotes();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:			openClinicalNotes();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:		openClinicalNotes();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:		openClinicalNotes();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:	private void openClinicalNotes()
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/demopatientreferrallist/GenForm.java:			fields[162] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedReferralRecordingVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/surgicalaudit/Logic.java:		FormMode plannedActualProcMode = form.lyrSurgAudit().tabPlannedActualProcedures().ccPlannedActualProc().getMode();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/surgicalaudit/Logic.java:		form.setMode(plannedActualProcMode);
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/extendedcareplanclinicalnotes/Logic.java:			openClinicalNote();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/extendedcareplanclinicalnotes/Logic.java:			openClinicalNote();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/extendedcareplanclinicalnotes/Logic.java:	private void openClinicalNote()
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/preceedingreferral/GenForm.java:			fields[162] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedPreceedingReferralVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/cliniclistwithicpactions/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/cliniclistwithicpactions/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/cliniclistwithicpactions/Logic.java:	protected void onCcListOwnerValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patientsummary/GenForm.java:		public void disableAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/patientsummary/GenForm.java:		public void hideAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/demoreferraldetails/GenForm.java:			fields[169] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedPreceedingReferralVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/demoreferraldetails/GenForm.java:			fields[191] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedReferralRecordingVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/Clinical/src/ims/clinical/forms/demoreferraldetails/GenForm.java:			fields[232] = new ims.framework.ReportField(this.context, prefix + "_lv_Clinical.DemoreferralDetails.__internal_x_context__SelectedReferral_" + componentIdentifier, "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.hbm.xml:<!-- debug: delayReasonGPUrgentReferral -->
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.hbm.xml:		<many-to-one name="delayReasonGPUrgentReferral" class="ims.domain.lookups.LookupInstance" access="field">
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	/** DelayReasonGPUrgentReferral */
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	private ims.domain.lookups.LookupInstance delayReasonGPUrgentReferral;
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	public ims.domain.lookups.LookupInstance getDelayReasonGPUrgentReferral() {
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		return delayReasonGPUrgentReferral;
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	public void setDelayReasonGPUrgentReferral(ims.domain.lookups.LookupInstance delayReasonGPUrgentReferral) {
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		this.delayReasonGPUrgentReferral = delayReasonGPUrgentReferral;
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		auditStr.append("\r\n*delayReasonGPUrgentReferral* :");
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		if (delayReasonGPUrgentReferral != null)
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			auditStr.append(delayReasonGPUrgentReferral.getText());
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		if (this.getDelayReasonGPUrgentReferral() != null)
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			sb.append("<delayReasonGPUrgentReferral>");
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			sb.append(this.getDelayReasonGPUrgentReferral().toXMLString()); 
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			sb.append("</delayReasonGPUrgentReferral>");		
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		fldEl = el.element("delayReasonGPUrgentReferral");
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			obj.setDelayReasonGPUrgentReferral(ims.domain.lookups.LookupInstance.fromXMLString(fldEl, factory)); 	
Source Library/openmaxims_workspace/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		public static final String DelayReasonGPUrgentReferral = "delayReasonGPUrgentReferral";
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/Logic.java:		form.getLocalContext().setPublicArea(form.getGlobalContext().Emergency.getTrackingPublicArea());
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GenForm.java:			throw new ims.framework.exceptions.CodingRuntimeException("The type 'Boolean' of the global context variable 'Emergency.TrackingPublicArea' is not supported.");
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GlobalContext.java:		public boolean getTrackingPublicAreaIsNotNull()
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GlobalContext.java:			return !cx_EmergencyTrackingPublicArea.getValueIsNull(context);
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GlobalContext.java:		public Boolean getTrackingPublicArea()
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GlobalContext.java:			return (Boolean)cx_EmergencyTrackingPublicArea.getValue(context);
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GlobalContext.java:		public void setTrackingPublicArea(Boolean value)
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GlobalContext.java:			cx_EmergencyTrackingPublicArea.setValue(context, value);
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/edwhiteboardnewdialog/GlobalContext.java:		private ims.framework.ContextVariable cx_EmergencyTrackingPublicArea = new ims.framework.ContextVariable("Emergency.TrackingPublicArea", "_cvp_Emergency.TrackingPublicArea");
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:		if (form.getGlobalContext().Emergency.getTrackingPublicArea()==null)
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:			form.getGlobalContext().Emergency.setTrackingPublicArea(false);
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if(noOfHospNumber != null && noOfHospNumber > 1 && !Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()))
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				else if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if(noOfHospNumber != null && noOfHospNumber > 1 && !Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()))
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if(noOfNHSNumber != null && noOfNHSNumber > 1 && !Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()))//WDEV-17966
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				else if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if(noOfNHSNumber != null && noOfNHSNumber > 1 && !Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()))
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:			if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && cell.getValue()!=null && !DynamicCellType.TABLE.equals(cell.getType()) && !DynamicCellType.BUTTON.equals(cell.getType()))
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:			else if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && DynamicCellType.BUTTON.equals(cell.getType()) && !(cell.getButtonText()==REFER_TO_BUTTON_TEXT || cell.getButtonText()==ALLOCATE_CUBICLE_BUTTON_TEXT  || cell.getButtonText()==SEEN_COMPLETE_BUTTON_TEXT))
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:		form.getGlobalContext().Emergency.setTrackingPublicArea(!form.getGlobalContext().Emergency.getTrackingPublicArea());
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/Logic.java:		form.btnPublicPrivateArea().setText(Boolean.FALSE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) ? "Public Area" : "Private Area" );
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GenForm.java:			throw new ims.framework.exceptions.CodingRuntimeException("The type 'Boolean' of the global context variable 'Emergency.TrackingPublicArea' is not supported.");
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		public boolean getTrackingPublicAreaIsNotNull()
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:			return !cx_EmergencyTrackingPublicArea.getValueIsNull(context);
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		public Boolean getTrackingPublicArea()
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:			return (Boolean)cx_EmergencyTrackingPublicArea.getValue(context);
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		public void setTrackingPublicArea(Boolean value)
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:			cx_EmergencyTrackingPublicArea.setValue(context, value);
Source Library/openmaxims_workspace/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		private ims.framework.ContextVariable cx_EmergencyTrackingPublicArea = new ims.framework.ContextVariable("Emergency.TrackingPublicArea", "_cvp_Emergency.TrackingPublicArea");
Source Library/openmaxims_workspace/Nursing/src/ims/nursing/forms/clinicaladmission/Logic.java:	private void openClinicalAdmission(ClinicalAdmissionVo voCA) 
Source Library/openmaxims_workspace/Nursing/src/ims/nursing/forms/clinicaladmission/Logic.java:			openClinicalAdmission(clinicalAdmission);
Source Library/openmaxims_workspace/Core/src/ims/core/forms/bedinfodialog/Handlers.java:	abstract protected void onBtnReOpenClick() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/bedinfodialog/Handlers.java:				onBtnReOpenClick();
Source Library/openmaxims_workspace/Core/src/ims/core/forms/bedinfodialog/Logic.java:	protected void onBtnReOpenClick() throws PresentationLogicException
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspupils/Logic.java:import ims.framework.utils.graphing.GraphingPupils;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspupils/Logic.java:		GraphingPupils point;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspupils/Logic.java:				point = new GraphingPupils(voVitalSign.getVitalsTakenDateTime(), 
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspupils/Logic.java:			GraphingPupils pointPupil= (GraphingPupils)point;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspulse/Logic.java:import ims.framework.utils.graphing.GraphingPulse;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspulse/Logic.java:		GraphingPulse point;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspulse/Logic.java:				point = new GraphingPulse(voVitalSign.getVitalsTakenDateTime(), voVitalSign.getPulse().getPulseRateRadial(), voVitalSign.getPulse().getPulseRateApex(), voVitalSign.getPulse().getIsIrregular(), voVitalSign);
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignspulse/Logic.java:			GraphingPulse pointPulse = (GraphingPulse) point;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/patientdocumentsview/Handlers.java:	abstract protected void onCcListDocumentsValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/patientdocumentsview/Handlers.java:				onCcListDocumentsValueChanged();
Source Library/openmaxims_workspace/Core/src/ims/core/forms/patientdocumentsview/Logic.java:	protected void onCcListDocumentsValueChanged() throws PresentationLogicException 
Source Library/openmaxims_workspace/Core/src/ims/core/forms/closeblockreopenbayorwarddlg/Handlers.java:	abstract protected void onBtnReOpenClick() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/closeblockreopenbayorwarddlg/Handlers.java:				onBtnReOpenClick();
Source Library/openmaxims_workspace/Core/src/ims/core/forms/closeblockreopenbayorwarddlg/Logic.java:	protected void onBtnReOpenClick() throws ims.framework.exceptions.PresentationLogicException
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignstprbp/Logic.java:import ims.framework.utils.graphing.GraphingPulse;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignstprbp/Logic.java:		GraphingPulse pointPulse;
Source Library/openmaxims_workspace/Core/src/ims/core/forms/vitalsignstprbp/Logic.java:					pointPulse = new GraphingPulse(voVitalSign.getVitalsTakenDateTime(),  voVitalSign.getPulse().getPulseRateRadial(), voVitalSign.getPulse().getPulseRateApex(), voVitalSign.getPulse().getIsIrregular(),voVitalSign);
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			cloneObject.DataCollection.get(index).Gpu = DataCollection.get(x).Gpu;
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:		if(Filter.Gpu != null && filter.Gpu.length()> 0)
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			filterString += "GPU" + ims.dto.NASMessageCodes.ATTRIBUTEVALUESEPARATOR + filter.Gpu;
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:		if(EditFilter.IncludeGpu)
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			dataString += "GPU" + ims.dto.NASMessageCodes.ATTRIBUTEVALUESEPARATOR + Connection.encodeFieldValue(data.Gpu);
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			record.Gpu = Connection.decodeFieldValue(Connection.getValueFor(valueItems, "GPU"));
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			record.Gpu = Connection.decodeFieldValue(Connection.getValueFor(valueItems, "GPU"));
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:		public String Gpu = "";
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			Gpu = "";
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:		public String Gpu = "";
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			Gpu = "";
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			newFilter.Gpu = this.Gpu;
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:		public boolean IncludeGpu = true;
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			IncludeGpu = true;
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			IncludeGpu = false;
Source Library/openmaxims_workspace/DTOClients/src/ims/dto/client/Outpat.java:			newEditFilter.IncludeGpu = this.IncludeGpu;
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/displacedappointmentsworklist/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/displacedappointmentsworklist/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/displacedappointmentsworklist/Logic.java:	protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/theatresessionmanagement/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/theatresessionmanagement/Handlers.java:	abstract protected void onBtnReOpenClick() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/theatresessionmanagement/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/theatresessionmanagement/Handlers.java:				onBtnReOpenClick();
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/theatresessionmanagement/Logic.java:	protected void onCcListOwnerValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/theatresessionmanagement/Logic.java:	protected void onBtnReOpenClick() throws PresentationLogicException
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:	abstract protected void onBtnReOpenClick() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:				onBtnReOpenClick();
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/sessionmanagement/Logic.java:	protected void onCcListOwnerValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace/Scheduling/src/ims/scheduling/forms/sessionmanagement/Logic.java:	protected void onBtnReOpenClick() throws PresentationLogicException
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:		form.lyrWaitingListDetails().tabPageMedical().GroupAdditionalProcedureMedical().setVisible(GroupAdditionalProcedureMedicalEnumeration.rdoSpecialtyHotListAdditionalProcMedical, showSecond);	//wdev-21151
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:		form.lyrWaitingListDetails().tabPageMedical().GroupAdditionalProcedureMedical().setVisible(GroupAdditionalProcedureMedicalEnumeration.rdoAllProceduresAdditionalProcMedical, showSecond);
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:				 form.lyrWaitingListDetails().tabPageMedical().GroupPrimaryProcedureMedical().setValue(GroupPrimaryProcedureMedicalEnumeration.rdoSpecialtyHotlistPrinaryProcMedical);			//wdev-21151
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:				 form.lyrWaitingListDetails().tabPageMedical().GroupAdditionalProcedureMedical().setValue(GroupAdditionalProcedureMedicalEnumeration.rdoSpecialtyHotListAdditionalProcMedical);	//wdev-21151
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:	private void addRowPrimaryProcMedical(ProcedureVo procedureVo)
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:	private void addRowSecondProcMedical(ProcedureVo procedureVo)
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:		form.lyrWaitingListDetails().tabPageMedical().GroupPrimaryProcedureMedical().setValue(GroupPrimaryProcedureMedicalEnumeration.rdoSpecialtyHotlistPrinaryProcMedical);		//wdev-21151
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:		form.lyrWaitingListDetails().tabPageMedical().GroupAdditionalProcedureMedical().setValue(GroupAdditionalProcedureMedicalEnumeration.rdoSpecialtyHotListAdditionalProcMedical);	//wdev-21151
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:			if( GroupAdditionalProcedureMedicalEnumeration.rdoSpecialtyHotListAdditionalProcMedical.equals(form.lyrWaitingListDetails().tabPageMedical().GroupAdditionalProcedureMedical().getValue())) 	//wdev-21151
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:			else if( GroupAdditionalProcedureMedicalEnumeration.rdoAllProceduresAdditionalProcMedical.equals(form.lyrWaitingListDetails().tabPageMedical().GroupAdditionalProcedureMedical().getValue()))
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:			addRowSecondProcMedical(procedureCollection.get(i));
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:			if( GroupPrimaryProcedureMedicalEnumeration.rdoSpecialtyHotlistPrinaryProcMedical.equals(form.lyrWaitingListDetails().tabPageMedical().GroupPrimaryProcedureMedical().getValue()))		//wdev-21151
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:			else if( GroupPrimaryProcedureMedicalEnumeration.rdoAllProceduresPrimaryProcMedical.equals(form.lyrWaitingListDetails().tabPageMedical().GroupPrimaryProcedureMedical().getValue()))
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/Logic.java:			addRowPrimaryProcMedical(procedureCollection.get(i));
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:						case 0: return GroupAdditionalProcedureMedicalEnumeration.rdoAllProceduresAdditionalProcMedical;
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:						case 1: return GroupAdditionalProcedureMedicalEnumeration.rdoSpecialtyHotListAdditionalProcMedical;
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:				public static GroupAdditionalProcedureMedicalEnumeration rdoAllProceduresAdditionalProcMedical = new GroupAdditionalProcedureMedicalEnumeration(0);
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:				public static GroupAdditionalProcedureMedicalEnumeration rdoSpecialtyHotListAdditionalProcMedical = new GroupAdditionalProcedureMedicalEnumeration(1);
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:						case 0: return GroupPrimaryProcedureMedicalEnumeration.rdoAllProceduresPrimaryProcMedical;
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:						case 1: return GroupPrimaryProcedureMedicalEnumeration.rdoSpecialtyHotlistPrinaryProcMedical;
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:				public static GroupPrimaryProcedureMedicalEnumeration rdoAllProceduresPrimaryProcMedical = new GroupPrimaryProcedureMedicalEnumeration(0);
Source Library/openmaxims_workspace/RefMan/src/ims/refman/forms/electivelistaddlaterdialog/GenForm.java:				public static GroupPrimaryProcedureMedicalEnumeration rdoSpecialtyHotlistPrinaryProcMedical = new GroupPrimaryProcedureMedicalEnumeration(1);
Source Library/openmaxims_workspace/RefMan/src/ims/RefMan/forms/transferelectivelistdialog/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/RefMan/src/ims/RefMan/forms/transferelectivelistdialog/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/RefMan/src/ims/RefMan/forms/transferelectivelistdialog/Logic.java:	protected void onCcListOwnerValueChanged()
Source Library/openmaxims_workspace/RefMan/src/ims/RefMan/forms/theatrelist/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace/RefMan/src/ims/RefMan/forms/theatrelist/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace/RefMan/src/ims/RefMan/forms/theatrelist/Logic.java:	protected void onCcListOwnerValueChanged() throws PresentationLogicException 
Source Library/openmaxims_workspace-archive/WebApp/MAXIMS_RefDataORA.sql:insert into applookup (id, name, description, active, hierarchical, systemtype) values( 1231059, 'DelayReasonGPUrgentReferral', 'DelayReasonGPUrgentReferral', 1, 0, 0)
Source Library/openmaxims_workspace-archive/WebApp/MAXIMS_RefDataORA.sql:insert into core_appcontextvari (id, vstp, rie, variablena, variableke, iscollecti, lkp_variablety, lkp_valuetype, valueclass) values( 1583, 0, 0, 'Emergency.TrackingPublicArea', '_cvp_Emergency.TrackingPublicArea', 0, -1178, -1183, 'Bool ')
Source Library/openmaxims_workspace-archive/WebApp/InsertLookupBoMapping.sql:insert into applookup_type_col_map(table_name, col_name, type_id, bo_name, bo_field) values('shcl_referralsrecor', 'lkp_delayreaso', 1231059, 'ReferralsRecording', 'delayReasonGPUrgentReferral')
Source Library/openmaxims_workspace-archive/WebApp/dictionary.dic:barracuda
Source Library/openmaxims_workspace-archive/WebApp/dictionary.dic:barracuda's
Source Library/openmaxims_workspace-archive/WebApp/dictionary.dic:barracudas
Source Library/openmaxims_workspace-archive/WebApp/MAXIMS_RefDataMSQ05.sql:insert into applookup (id, name, description, active, hierarchical, systemtype) values( 1231059, 'DelayReasonGPUrgentReferral', 'DelayReasonGPUrgentReferral', 1, 0, 0)
Source Library/openmaxims_workspace-archive/WebApp/MAXIMS_RefDataMSQ05.sql:insert into core_appcontextvari (id, vstp, rie, variablena, variableke, iscollecti, lkp_variablety, lkp_valuetype, valueclass) values( 1583, 0, 0, 'Emergency.TrackingPublicArea', '_cvp_Emergency.TrackingPublicArea', 0, -1178, -1183, 'Bool')
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:		if(this.delayreasongpurgentreferral == null)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:			clone.delayreasongpurgentreferral = null;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingVo.java:			clone.delayreasongpurgentreferral = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)this.delayreasongpurgentreferral.clone();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:	public ims.vo.LookupInstanceBean getDelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		return this.delayreasongpurgentreferral;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:	public void setDelayReasonGPUrgentReferral(ims.vo.LookupInstanceBean value)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:		this.delayreasongpurgentreferral = value;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingShortVoBean.java:	private ims.vo.LookupInstanceBean delayreasongpurgentreferral;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		this.delayreasongpurgentreferral = vo.getDelayReasonGPUrgentReferral() == null ? null : (ims.vo.LookupInstanceBean)vo.getDelayReasonGPUrgentReferral().getBean();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:	public ims.vo.LookupInstanceBean getDelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		return this.delayreasongpurgentreferral;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:	public void setDelayReasonGPUrgentReferral(ims.vo.LookupInstanceBean value)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:		this.delayreasongpurgentreferral = value;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/beans/ReferralsRecordingVoBean.java:	private ims.vo.LookupInstanceBean delayreasongpurgentreferral;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		this.delayreasongpurgentreferral = bean.getDelayReasonGPUrgentReferral() == null ? null : ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.buildLookup(bean.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		if(fieldName.equals("DELAYREASONGPURGENTREFERRAL"))
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:			return getDelayReasonGPUrgentReferral();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	public boolean getDelayReasonGPUrgentReferralIsNotNull()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		return this.delayreasongpurgentreferral != null;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	public ims.clinical.vo.lookups.DelayReasonGPUrgentReferral getDelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		return this.delayreasongpurgentreferral;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	public void setDelayReasonGPUrgentReferral(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		this.delayreasongpurgentreferral = value;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		if(this.delayreasongpurgentreferral == null)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:			clone.delayreasongpurgentreferral = null;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:			clone.delayreasongpurgentreferral = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)this.delayreasongpurgentreferral.clone();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:		if(this.delayreasongpurgentreferral != null)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/ReferralsRecordingShortVo.java:	protected ims.clinical.vo.lookups.DelayReasonGPUrgentReferral delayreasongpurgentreferral;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:public class DelayReasonGPUrgentReferralCollection extends LookupInstanceCollection implements ims.vo.ImsCloneable, TreeModel
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public void add(DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public int indexOf(DelayReasonGPUrgentReferral instance)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public boolean contains(DelayReasonGPUrgentReferral instance)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public DelayReasonGPUrgentReferral get(int index)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		return (DelayReasonGPUrgentReferral)super.getIndex(index);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public void remove(DelayReasonGPUrgentReferral instance)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferralCollection newCol = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferral item;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:			newCol.add(new DelayReasonGPUrgentReferral(item.getID(), item.getText(), item.isActive(), item.getParent(), item.getImage(), item.getColor(), item.getOrder()));
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:					item.setParent((DelayReasonGPUrgentReferral)item.getParent().clone());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public DelayReasonGPUrgentReferral getInstance(int instanceId)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		return (DelayReasonGPUrgentReferral)super.getInstanceById(instanceId);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public DelayReasonGPUrgentReferral[] toArray()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferral[] arr = new DelayReasonGPUrgentReferral[this.size()];
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public static DelayReasonGPUrgentReferralCollection buildFromBeanCollection(java.util.Collection beans)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferralCollection coll = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:			coll.add(DelayReasonGPUrgentReferral.buildLookup((ims.vo.LookupInstanceBean)iter.next()));
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:	public static DelayReasonGPUrgentReferralCollection buildFromBeanCollection(ims.vo.LookupInstanceBean[] beans)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:		DelayReasonGPUrgentReferralCollection coll = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferralCollection.java:			coll.add(DelayReasonGPUrgentReferral.buildLookup(beans[x]));
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:	public static DelayReasonGPUrgentReferralCollection getDelayReasonGPUrgentReferral(LookupService lookupService) {
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:		DelayReasonGPUrgentReferralCollection collection =
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:			(DelayReasonGPUrgentReferralCollection) lookupService.getLookupCollection(DelayReasonGPUrgentReferral.TYPE_ID, 
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:				DelayReasonGPUrgentReferralCollection.class, DelayReasonGPUrgentReferral.class);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:	public static DelayReasonGPUrgentReferral getDelayReasonGPUrgentReferralInstance(LookupService lookupService, int id) 
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/LookupHelper.java:		return (DelayReasonGPUrgentReferral)lookupService.getLookupInstance(DelayReasonGPUrgentReferral.class, DelayReasonGPUrgentReferral.TYPE_ID, id);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:public class DelayReasonGPUrgentReferral extends ims.vo.LookupInstVo implements TreeNode
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active, DelayReasonGPUrgentReferral parent, Image image)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active, DelayReasonGPUrgentReferral parent, Image image, Color color)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral(int id, String text, boolean active, DelayReasonGPUrgentReferral parent, Image image, Color color, int order)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral buildLookup(ims.vo.LookupInstanceBean bean)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return new DelayReasonGPUrgentReferral(bean.getId(), bean.getText(), bean.isActive());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return (DelayReasonGPUrgentReferral)super.getParentInstance();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public DelayReasonGPUrgentReferral getParent()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return (DelayReasonGPUrgentReferral)super.getParentInstance();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public void setParent(DelayReasonGPUrgentReferral parent)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		DelayReasonGPUrgentReferral[] typedChildren = new DelayReasonGPUrgentReferral[children.size()];
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:			typedChildren[i] = (DelayReasonGPUrgentReferral)children.get(i);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		if (child instanceof DelayReasonGPUrgentReferral)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:			super.addChild((DelayReasonGPUrgentReferral)child);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		if (child instanceof DelayReasonGPUrgentReferral)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:			super.removeChild((DelayReasonGPUrgentReferral)child);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		DelayReasonGPUrgentReferralCollection result = new DelayReasonGPUrgentReferralCollection();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral[] getNegativeInstances()
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:		return new DelayReasonGPUrgentReferral[] {};
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral getNegativeInstance(String name)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/lookups/DelayReasonGPUrgentReferral.java:	public static DelayReasonGPUrgentReferral getNegativeInstance(Integer id)
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		valueObjectDest.setDelayReasonGPUrgentReferral(valueObjectSrc.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:				// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		ims.domain.lookups.LookupInstance instance29 = domainObject.getDelayReasonGPUrgentReferral();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral voLookup29 = new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(instance29.getId(),instance29.getText(), instance29.isActive(), null, img, color);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral parentVoLookup29 = voLookup29;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:								parentVoLookup29.setParent(new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(parent29.getId(),parent29.getText(), parent29.isActive(), null, img, color));
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:			valueObject.setDelayReasonGPUrgentReferral(voLookup29);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		if ( null != valueObject.getDelayReasonGPUrgentReferral() ) 
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:				domainFactory.getLookupInstance(valueObject.getDelayReasonGPUrgentReferral().getID());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingVoAssembler.java:		domainObject.setDelayReasonGPUrgentReferral(value29);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		valueObjectDest.setDelayReasonGPUrgentReferral(valueObjectSrc.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:				// DelayReasonGPUrgentReferral
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		ims.domain.lookups.LookupInstance instance25 = domainObject.getDelayReasonGPUrgentReferral();
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral voLookup25 = new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(instance25.getId(),instance25.getText(), instance25.isActive(), null, img, color);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral parentVoLookup25 = voLookup25;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:								parentVoLookup25.setParent(new ims.clinical.vo.lookups.DelayReasonGPUrgentReferral(parent25.getId(),parent25.getText(), parent25.isActive(), null, img, color));
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:			valueObject.setDelayReasonGPUrgentReferral(voLookup25);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		if ( null != valueObject.getDelayReasonGPUrgentReferral() ) 
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:				domainFactory.getLookupInstance(valueObject.getDelayReasonGPUrgentReferral().getID());
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/clinical/vo/domain/ReferralsRecordingShortVoAssembler.java:		domainObject.setDelayReasonGPUrgentReferral(value25);
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/vo/lookups/ClassHelper.java:				return ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.class;
Source Library/openmaxims_workspace-archive/ValueObjects/src/ims/vo/lookups/ClassHelper.java:				return ims.clinical.vo.lookups.DelayReasonGPUrgentReferralCollection.class;
Source Library/openmaxims_workspace-archive/ICP/src/ims/icp/forms/patienticpactiondetails/Handlers.java:	abstract protected void onCcLinkedActionDetailsValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace-archive/ICP/src/ims/icp/forms/patienticpactiondetails/Handlers.java:				onCcLinkedActionDetailsValueChanged();
Source Library/openmaxims_workspace-archive/ICP/src/ims/icp/forms/patienticpactiondetails/Logic.java:	protected void onCcLinkedActionDetailsValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/Logic.java:		form.cmbDelayReasonFirstSeen().setValue(rrVo.getDelayReasonGPUrgentReferral());
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/Logic.java:		rrVo.setDelayReasonGPUrgentReferral(form.cmbDelayReasonFirstSeen().getValue());
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text)
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text, ims.framework.utils.Image image)
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text, ims.framework.utils.Color textColor)
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void newRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value, String text, ims.framework.utils.Image image, ims.framework.utils.Color textColor)
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public boolean removeRow(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public ims.clinical.vo.lookups.DelayReasonGPUrgentReferral getValue()
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:			return (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)super.control.getValue();
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:		public void setValue(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral value)
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:			fields[89] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedPreceedingReferralVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/GenForm.java:			fields[111] = new ims.framework.ReportField(this.context, prefix + "_lv_Oncology.CancerreferralDetails.__internal_x_context__SelectedReferral_" + componentIdentifier, "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:					ims.clinical.vo.lookups.DelayReasonGPUrgentReferral existingInstance = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)listOfValues.get(x);
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		else if(value instanceof ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:			ims.clinical.vo.lookups.DelayReasonGPUrgentReferral instance = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)value;
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:					ims.clinical.vo.lookups.DelayReasonGPUrgentReferral existingInstance = (ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)listOfValues.get(x);
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		ims.clinical.vo.lookups.DelayReasonGPUrgentReferralCollection lookupCollection = ims.clinical.vo.lookups.LookupHelper.getDelayReasonGPUrgentReferral(this.domain.getLookupService());
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		ims.clinical.vo.lookups.DelayReasonGPUrgentReferral instance = ims.clinical.vo.lookups.LookupHelper.getDelayReasonGPUrgentReferralInstance(this.domain.getLookupService(), id);
Source Library/openmaxims_workspace-archive/Oncology/src/ims/oncology/forms/cancerreferraldetails/BaseLogic.java:		this.form.cmbDelayReasonFirstSeen().setValue((ims.clinical.vo.lookups.DelayReasonGPUrgentReferral)domain.getLookupService().getDefaultInstance(ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.class, engine.getFormName().getID(), ims.clinical.vo.lookups.DelayReasonGPUrgentReferral.TYPE_ID));
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/referralsrecording/GenForm.java:			fields[89] = new ims.framework.ReportField(this.context, prefix + "_lv_Clinical.ReferralsRecording.__internal_x_context__EditedRecord_" + componentIdentifier, "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/surgicalauditpreoperationchecks/GenForm.java:	public TextBox txtProcMandatory()
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/demoreferrallist/GenForm.java:			fields[89] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedReferralRecordingVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/patienticplist/Logic.java:				openCloseICP();//WDEV-12965
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/patienticplist/Logic.java:				openCloseICP();//WDEV-12965
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/patienticplist/Logic.java:	private void openCloseICP()//WDEV-12965
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/patient_summary/GenForm.java:		public void disableAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/patient_summary/GenForm.java:		public void hideAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:					openClinicalNotes();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:					openClinicalNotes();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:			openClinicalNotes();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:		openClinicalNotes();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:		openClinicalNotes();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/clinicalnotedrawing/Logic.java:	private void openClinicalNotes()
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/demopatientreferrallist/GenForm.java:			fields[89] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedReferralRecordingVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/surgicalaudit/Logic.java:		FormMode plannedActualProcMode = form.lyrSurgAudit().tabPlannedActualProcedures().ccPlannedActualProc().getMode();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/surgicalaudit/Logic.java:		form.setMode(plannedActualProcMode);
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/extendedcareplanclinicalnotes/Logic.java:			openClinicalNote();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/extendedcareplanclinicalnotes/Logic.java:			openClinicalNote();
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/extendedcareplanclinicalnotes/Logic.java:	private void openClinicalNote()
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/preceedingreferral/GenForm.java:			fields[89] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedPreceedingReferralVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/patientsummary/GenForm.java:		public void disableAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/patientsummary/GenForm.java:		public void hideAllPatSummaryProcMenuItems()
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/demoreferraldetails/GenForm.java:			fields[96] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedPreceedingReferralVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/demoreferraldetails/GenForm.java:			fields[118] = new ims.framework.ReportField(this.context, "_cv_Clinical.SelectedReferralRecordingVo", "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/Clinical/src/ims/clinical/forms/demoreferraldetails/GenForm.java:			fields[159] = new ims.framework.ReportField(this.context, prefix + "_lv_Clinical.DemoreferralDetails.__internal_x_context__SelectedReferral_" + componentIdentifier, "BO-1072100062-DELAYREASONGPURGENTREFERRAL", "DelayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.hbm.xml:<!-- debug: delayReasonGPUrgentReferral -->
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.hbm.xml:		<many-to-one name="delayReasonGPUrgentReferral" class="ims.domain.lookups.LookupInstance" access="field">
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	/** DelayReasonGPUrgentReferral */
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	private ims.domain.lookups.LookupInstance delayReasonGPUrgentReferral;
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	public ims.domain.lookups.LookupInstance getDelayReasonGPUrgentReferral() {
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		return delayReasonGPUrgentReferral;
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:	public void setDelayReasonGPUrgentReferral(ims.domain.lookups.LookupInstance delayReasonGPUrgentReferral) {
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		this.delayReasonGPUrgentReferral = delayReasonGPUrgentReferral;
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		auditStr.append("\r\n*delayReasonGPUrgentReferral* :");
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		if (delayReasonGPUrgentReferral != null)
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			auditStr.append(delayReasonGPUrgentReferral.getText());
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		if (this.getDelayReasonGPUrgentReferral() != null)
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			sb.append("<delayReasonGPUrgentReferral>");
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			sb.append(this.getDelayReasonGPUrgentReferral().toXMLString()); 
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			sb.append("</delayReasonGPUrgentReferral>");		
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		fldEl = el.element("delayReasonGPUrgentReferral");
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:			obj.setDelayReasonGPUrgentReferral(ims.domain.lookups.LookupInstance.fromXMLString(fldEl, factory)); 	
Source Library/openmaxims_workspace-archive/DomainObjects/src/ims/clinical/domain/objects/ReferralsRecording.java:		public static final String DelayReasonGPUrgentReferral = "delayReasonGPUrgentReferral";
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:		if (form.getGlobalContext().Emergency.getTrackingPublicArea()==null)
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:			form.getGlobalContext().Emergency.setTrackingPublicArea(false);
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:				if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && tableCell.getValue()!=null )
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:			if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && cell.getValue()!=null && !DynamicCellType.TABLE.equals(cell.getType()) && !DynamicCellType.BUTTON.equals(cell.getType()))
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:			else if (maskInPublicArea(column) && Boolean.TRUE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) && DynamicCellType.BUTTON.equals(cell.getType()) && !(cell.getButtonText()==REFER_TO_BUTTON_TEXT || cell.getButtonText()==ALLOCATE_CUBICLE_BUTTON_TEXT  || cell.getButtonText()==SEEN_COMPLETE_BUTTON_TEXT))
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:		form.getGlobalContext().Emergency.setTrackingPublicArea(!form.getGlobalContext().Emergency.getTrackingPublicArea());
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/Logic.java:		form.btnPublicPrivateArea().setText(Boolean.FALSE.equals(form.getGlobalContext().Emergency.getTrackingPublicArea()) ? "Public Area" : "Private Area" );
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GenForm.java:			throw new ims.framework.exceptions.CodingRuntimeException("The type 'Boolean' of the global context variable 'Emergency.TrackingPublicArea' is not supported.");
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		public boolean getTrackingPublicAreaIsNotNull()
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:			return !cx_EmergencyTrackingPublicArea.getValueIsNull(context);
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		public Boolean getTrackingPublicArea()
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:			return (Boolean)cx_EmergencyTrackingPublicArea.getValue(context);
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		public void setTrackingPublicArea(Boolean value)
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:			cx_EmergencyTrackingPublicArea.setValue(context, value);
Source Library/openmaxims_workspace-archive/Emergency/src/ims/emergency/forms/tracking/GlobalContext.java:		private ims.framework.ContextVariable cx_EmergencyTrackingPublicArea = new ims.framework.ContextVariable("Emergency.TrackingPublicArea", "_cvp_Emergency.TrackingPublicArea");
Source Library/openmaxims_workspace-archive/Nursing/src/ims/nursing/forms/clinicaladmission/Logic.java:	private void openClinicalAdmission(ClinicalAdmissionVo voCA) 
Source Library/openmaxims_workspace-archive/Nursing/src/ims/nursing/forms/clinicaladmission/Logic.java:			openClinicalAdmission(clinicalAdmission);
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/bedinfodialog/Handlers.java:	abstract protected void onBtnReOpenClick() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/bedinfodialog/Handlers.java:				onBtnReOpenClick();
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/bedinfodialog/Logic.java:	protected void onBtnReOpenClick() throws PresentationLogicException
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspupils/Logic.java:import ims.framework.utils.graphing.GraphingPupils;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspupils/Logic.java:		GraphingPupils point;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspupils/Logic.java:				point = new GraphingPupils(voVitalSign.getVitalsTakenDateTime(), 
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspupils/Logic.java:			GraphingPupils pointPupil= (GraphingPupils)point;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspulse/Logic.java:import ims.framework.utils.graphing.GraphingPulse;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspulse/Logic.java:		GraphingPulse point;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspulse/Logic.java:				point = new GraphingPulse(voVitalSign.getVitalsTakenDateTime(), voVitalSign.getPulse().getPulseRateRadial(), voVitalSign.getPulse().getPulseRateApex(), voVitalSign.getPulse().getIsIrregular(), voVitalSign);
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignspulse/Logic.java:			GraphingPulse pointPulse = (GraphingPulse) point;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/patientdocumentsview/Handlers.java:	abstract protected void onCcListDocumentsValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/patientdocumentsview/Handlers.java:				onCcListDocumentsValueChanged();
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/patientdocumentsview/Logic.java:	protected void onCcListDocumentsValueChanged() throws PresentationLogicException 
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignstprbp/Logic.java:import ims.framework.utils.graphing.GraphingPulse;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignstprbp/Logic.java:		GraphingPulse pointPulse;
Source Library/openmaxims_workspace-archive/Core/src/ims/core/forms/vitalsignstprbp/Logic.java:					pointPulse = new GraphingPulse(voVitalSign.getVitalsTakenDateTime(),  voVitalSign.getPulse().getPulseRateRadial(), voVitalSign.getPulse().getPulseRateApex(), voVitalSign.getPulse().getIsIrregular(),voVitalSign);
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			cloneObject.DataCollection.get(index).Gpu = DataCollection.get(x).Gpu;
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:		if(Filter.Gpu != null && filter.Gpu.length()> 0)
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			filterString += "GPU" + ims.dto.NASMessageCodes.ATTRIBUTEVALUESEPARATOR + filter.Gpu;
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:		if(EditFilter.IncludeGpu)
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			dataString += "GPU" + ims.dto.NASMessageCodes.ATTRIBUTEVALUESEPARATOR + Connection.encodeFieldValue(data.Gpu);
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			record.Gpu = Connection.decodeFieldValue(Connection.getValueFor(valueItems, "GPU"));
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			record.Gpu = Connection.decodeFieldValue(Connection.getValueFor(valueItems, "GPU"));
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:		public String Gpu = "";
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			Gpu = "";
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:		public String Gpu = "";
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			Gpu = "";
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			newFilter.Gpu = this.Gpu;
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:		public boolean IncludeGpu = true;
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			IncludeGpu = true;
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			IncludeGpu = false;
Source Library/openmaxims_workspace-archive/DTOClients/src/ims/dto/client/Outpat.java:			newEditFilter.IncludeGpu = this.IncludeGpu;
Source Library/openmaxims_workspace-archive/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace-archive/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:	abstract protected void onBtnReOpenClick() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace-archive/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace-archive/Scheduling/src/ims/scheduling/forms/sessionmanagement/Handlers.java:				onBtnReOpenClick();
Source Library/openmaxims_workspace-archive/Scheduling/src/ims/scheduling/forms/sessionmanagement/Logic.java:	protected void onCcListOwnerValueChanged() throws PresentationLogicException
Source Library/openmaxims_workspace-archive/Scheduling/src/ims/scheduling/forms/sessionmanagement/Logic.java:	protected void onBtnReOpenClick() throws PresentationLogicException
Source Library/openmaxims_workspace-archive/RefMan/src/ims/RefMan/forms/transferelectivelistdialog/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace-archive/RefMan/src/ims/RefMan/forms/transferelectivelistdialog/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace-archive/RefMan/src/ims/RefMan/forms/transferelectivelistdialog/Logic.java:	protected void onCcListOwnerValueChanged()
Source Library/openmaxims_workspace-archive/RefMan/src/ims/RefMan/forms/theatrelist/Handlers.java:	abstract protected void onCcListOwnerValueChanged() throws ims.framework.exceptions.PresentationLogicException;
Source Library/openmaxims_workspace-archive/RefMan/src/ims/RefMan/forms/theatrelist/Handlers.java:				onCcListOwnerValueChanged();
Source Library/openmaxims_workspace-archive/RefMan/src/ims/RefMan/forms/theatrelist/Logic.java:	protected void onCcListOwnerValueChanged() throws PresentationLogicException 

```
