# https://github.com/Starlink/ORAC-DR

```console
src/lib/perl5/ORAC/Calib/JCMTCont.pm:    my $scudate = $self->thing->{'ORACUT'}; # the thing method is the header
src/lib/perl5/ORAC/Calib/JCMTCont.pm:    if (defined $scudate) {
src/lib/perl5/ORAC/Calib/JCMTCont.pm:      my $y = substr($scudate, 2, 2);
src/lib/perl5/ORAC/Calib/JCMTCont.pm:      my $m = substr($scudate, 4, 2);
src/lib/perl5/ORAC/Calib/JCMTCont.pm:      my $d = substr($scudate, 6, 2);
src/lib/perl5/ORAC/Calib/JCMTCont.pm:      $scudate = "$d $m $y";
src/lib/perl5/ORAC/Calib/JCMTCont.pm:      $scudate = '1 1 1';
src/lib/perl5/ORAC/Calib/JCMTCont.pm:    my $status = $self->fluxes_mon->obeyw("","$hidden planet=$source date='$scudate' time='$scutime' filter=$filter");
src/lib/perl5/ORAC/Calib/JCMTCont.pm:    orac_debug "Called fluxes with $hidden planet=$source date='$scudate' time='$scutime' filter=$filter. \n";

```
