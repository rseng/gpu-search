# https://github.com/luinix/pesco

```console
pesco/ejercicios/gui/PodiumPanel.cs:		Cairo.Context cairoCMedal;
pesco/ejercicios/gui/PodiumPanel.cs:				if ( cairoCMedal != null )
pesco/ejercicios/gui/PodiumPanel.cs:					((IDisposable) cairoCMedal).Dispose(); 
pesco/ejercicios/gui/PodiumPanel.cs:			cairoCMedal = Gdk.CairoHelper.Create ( imageBackground.Pixmap );
pesco/ejercicios/gui/PodiumPanel.cs:				cairoCMedal.Matrix = new Matrix();
pesco/ejercicios/gui/PodiumPanel.cs:				cairoCMedal.Translate( 442+((256.0-(256.0*auxFactor))/2), 222+((256.0-(256.0*auxFactor))/2) );
pesco/ejercicios/gui/PodiumPanel.cs:				cairoCMedal.Scale( auxFactor, auxFactor );								
pesco/ejercicios/gui/PodiumPanel.cs:				CairoHelper.SetSourcePixbuf (cairoCMedal, medalPixbuf, 0, 0);
pesco/ejercicios/gui/PodiumPanel.cs:				cairoCMedal.PaintWithAlpha (auxAlpha);
pesco/ejercicios/gui/TotalPodiumPanel.cs:		Cairo.Context cairoCMedal;
pesco/ejercicios/gui/TotalPodiumPanel.cs:			if ( cairoCMedal != null )
pesco/ejercicios/gui/TotalPodiumPanel.cs:				((IDisposable) cairoCMedal).Dispose(); 
pesco/ejercicios/gui/TotalPodiumPanel.cs:				if ( cairoCMedal != null )
pesco/ejercicios/gui/TotalPodiumPanel.cs:					((IDisposable) cairoCMedal).Dispose(); 
pesco/ejercicios/gui/TotalPodiumPanel.cs:			cairoCMedal = Gdk.CairoHelper.Create ( imageBackground.Pixmap );
pesco/ejercicios/gui/TotalPodiumPanel.cs:				CairoHelper.SetSourcePixbuf (cairoCMedal, medalPixbuf, 0, 0);
pesco/ejercicios/gui/TotalPodiumPanel.cs:				cairoCMedal.Matrix = new Matrix();					                           
pesco/ejercicios/gui/TotalPodiumPanel.cs:				cairoCMedal.Translate( 442+((256.0-(256.0*auxFactor))/2), 222+((256.0-(256.0*auxFactor))/2) );
pesco/ejercicios/gui/TotalPodiumPanel.cs:				cairoCMedal.Scale( auxFactor, auxFactor );
pesco/ejercicios/gui/TotalPodiumPanel.cs:				cairoCMedal.PaintWithAlpha (auxAlpha);
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    buttonc1.Clicked+=new EventHandler(OnButtoncClicked );
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    eventboxc1.ButtonPressEvent+=new ButtonPressEventHandler(OnButtoncClicked);
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    buttonc2.Clicked+=new EventHandler(OnButtoncClicked );
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    eventboxc2.ButtonPressEvent+=new ButtonPressEventHandler(OnButtoncClicked);
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    buttonc3.Clicked+=new EventHandler(OnButtoncClicked );
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    eventboxc3.ButtonPressEvent+=new ButtonPressEventHandler(OnButtoncClicked);
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    buttonc4.Clicked+=new EventHandler(OnButtoncClicked );
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:                    eventboxc4.ButtonPressEvent+=new ButtonPressEventHandler(OnButtoncClicked);
pesco/ejercicios/ObjetosClasificables/gui/TrialPanel.cs:  protected virtual void OnButtoncClicked (object sender, System.EventArgs e)
pesco/gtk-gui/gui.stetic:                                            <signal name="Clicked" handler="OnButtoncClicked" />
pesco/gtk-gui/gui.stetic:                                            <signal name="Clicked" handler="OnButtoncClicked" />
pesco/gtk-gui/gui.stetic:                                            <signal name="Clicked" handler="OnButtoncClicked" />
pesco/gtk-gui/gui.stetic:                                            <signal name="Clicked" handler="OnButtoncClicked" />
pesco/gtk-gui/pesco.TrialPanel.cs:			this.buttonc1.Clicked += new global::System.EventHandler (this.OnButtoncClicked);
pesco/gtk-gui/pesco.TrialPanel.cs:			this.buttonc2.Clicked += new global::System.EventHandler (this.OnButtoncClicked);
pesco/gtk-gui/pesco.TrialPanel.cs:			this.buttonc3.Clicked += new global::System.EventHandler (this.OnButtoncClicked);
pesco/gtk-gui/pesco.TrialPanel.cs:			this.buttonc4.Clicked += new global::System.EventHandler (this.OnButtoncClicked);

```
