# https://github.com/chapmanb/bcbb

```console
semantic/biomart.py:        builder = BioMartQueryBuilder("snp", "jpNCCLiver")
semantic/biomart.py:FROM dataset:snp_jpNCCLiver
talks/reveal.js/css/reveal.css: 				background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABkklEQVRYR8WX4VHDMAxG6wnoJrABZQPYBCaBTWAD2g1gE5gg6OOsXuxIlr40d81dfrSJ9V4c2VLK7spHuTJ/5wpM07QXuXc5X0opX2tEJcadjHuV80li/FgxTIEK/5QBCICBD6xEhSMGHgQPgBgLiYVAB1dpSqKDawxTohFw4JSEA3clzgIBPCURwE2JucBR7rhPJJv5OpJwDX+SfDjgx1wACQeJG1aChP9K/IMmdZ8DtESV1WyP3Bt4MwM6sj4NMxMYiqUWHQu4KYA/SYkIjOsm3BXYWMKFDwU2khjCQ4ELJUJ4SmClRArOCmSXGuKma0fYD5CbzHxFpCSGAhfAVSSUGDUk2BWZaff2g6GE15BsBQ9nwmpIGDiyHQddwNTMKkbZaf9fajXQca1EX44puJZUsnY0ObGmITE3GVLCbEhQUjGVt146j6oasWN+49Vph2w1pZ5EansNZqKBm1txbU57iRRcZ86RWMDdWtBJUHBHwoQPi1GV+JCbntmvok7iTX4/Up9mgyTc/FJYDTcndgH/AA5A/CHsyEkVAAAAAElFTkSuQmCC);
talks/reveal.js/css/reveal.min.css: */ html,body,.reveal div,.reveal span,.reveal applet,.reveal object,.reveal iframe,.reveal h1,.reveal h2,.reveal h3,.reveal h4,.reveal h5,.reveal h6,.reveal p,.reveal blockquote,.reveal pre,.reveal a,.reveal abbr,.reveal acronym,.reveal address,.reveal big,.reveal cite,.reveal code,.reveal del,.reveal dfn,.reveal em,.reveal img,.reveal ins,.reveal kbd,.reveal q,.reveal s,.reveal samp,.reveal small,.reveal strike,.reveal strong,.reveal sub,.reveal sup,.reveal tt,.reveal var,.reveal b,.reveal u,.reveal i,.reveal center,.reveal dl,.reveal dt,.reveal dd,.reveal ol,.reveal ul,.reveal li,.reveal fieldset,.reveal form,.reveal label,.reveal legend,.reveal table,.reveal caption,.reveal tbody,.reveal tfoot,.reveal thead,.reveal tr,.reveal th,.reveal td,.reveal article,.reveal aside,.reveal canvas,.reveal details,.reveal embed,.reveal figure,.reveal figcaption,.reveal footer,.reveal header,.reveal hgroup,.reveal menu,.reveal nav,.reveal output,.reveal ruby,.reveal section,.reveal summary,.reveal time,.reveal mark,.reveal audio,video{margin:0;padding:0;border:0;font-size:100%;font:inherit;vertical-align:baseline}.reveal article,.reveal aside,.reveal details,.reveal figcaption,.reveal figure,.reveal footer,.reveal header,.reveal hgroup,.reveal menu,.reveal nav,.reveal section{display:block}html,body{width:100%;height:100%;overflow:hidden}body{position:relative;line-height:1}::selection{background:#FF5E99;color:#fff;text-shadow:none}.reveal h1,.reveal h2,.reveal h3,.reveal h4,.reveal h5,.reveal h6{-webkit-hyphens:auto;-moz-hyphens:auto;hyphens:auto;word-wrap:break-word}.reveal h1{font-size:3.77em}.reveal h2{font-size:2.11em}.reveal h3{font-size:1.55em}.reveal h4{font-size:1em}.reveal .slides section .fragment{opacity:0;-webkit-transition:all .2s ease;-moz-transition:all .2s ease;-ms-transition:all .2s ease;-o-transition:all .2s ease;transition:all .2s ease}.reveal .slides section .fragment.visible{opacity:1}.reveal .slides section .fragment.grow{opacity:1}.reveal .slides section .fragment.grow.visible{-webkit-transform:scale(1.3);-moz-transform:scale(1.3);-ms-transform:scale(1.3);-o-transform:scale(1.3);transform:scale(1.3)}.reveal .slides section .fragment.shrink{opacity:1}.reveal .slides section .fragment.shrink.visible{-webkit-transform:scale(0.7);-moz-transform:scale(0.7);-ms-transform:scale(0.7);-o-transform:scale(0.7);transform:scale(0.7)}.reveal .slides section .fragment.zoom-in{opacity:0;-webkit-transform:scale(0.1);-moz-transform:scale(0.1);-ms-transform:scale(0.1);-o-transform:scale(0.1);transform:scale(0.1)}.reveal .slides section .fragment.zoom-in.visible{opacity:1;-webkit-transform:scale(1);-moz-transform:scale(1);-ms-transform:scale(1);-o-transform:scale(1);transform:scale(1)}.reveal .slides section .fragment.roll-in{opacity:0;-webkit-transform:rotateX(90deg);-moz-transform:rotateX(90deg);-ms-transform:rotateX(90deg);-o-transform:rotateX(90deg);transform:rotateX(90deg)}.reveal .slides section .fragment.roll-in.visible{opacity:1;-webkit-transform:rotateX(0);-moz-transform:rotateX(0);-ms-transform:rotateX(0);-o-transform:rotateX(0);transform:rotateX(0)}.reveal .slides section .fragment.fade-out{opacity:1}.reveal .slides section .fragment.fade-out.visible{opacity:0}.reveal .slides section .fragment.semi-fade-out{opacity:1}.reveal .slides section .fragment.semi-fade-out.visible{opacity:.5}.reveal .slides section .fragment.highlight-red,.reveal .slides section .fragment.highlight-green,.reveal .slides section .fragment.highlight-blue{opacity:1}.reveal .slides section .fragment.highlight-red.visible{color:#ff2c2d}.reveal .slides section .fragment.highlight-green.visible{color:#17ff2e}.reveal .slides section .fragment.highlight-blue.visible{color:#1b91ff}.reveal:after{content:'';font-style:italic}.reveal img,.reveal video,.reveal iframe{max-width:95%;max-height:95%}.reveal a{position:relative}.reveal strong,.reveal b{font-weight:700}.reveal em,.reveal i{font-style:italic}.reveal ol,.reveal ul{display:inline-block;text-align:left;margin:0 0 0 1em}.reveal ol{list-style-type:decimal}.reveal ul{list-style-type:disc}.reveal ul ul{list-style-type:square}.reveal ul ul ul{list-style-type:circle}.reveal ul ul,.reveal ul ol,.reveal ol ol,.reveal ol ul{display:block;margin-left:40px}.reveal p{margin-bottom:10px;line-height:1.2em}.reveal q,.reveal blockquote{quotes:none}.reveal blockquote{display:block;position:relative;width:70%;margin:5px auto;padding:5px;font-style:italic;background:rgba(255,255,255,.05);box-shadow:0 0 2px rgba(0,0,0,.2)}.reveal blockquote p:first-child,.reveal blockquote p:last-child{display:inline-block}.reveal q{font-style:italic}.reveal pre{display:block;position:relative;width:90%;margin:15px auto;text-align:left;font-size:.55em;font-family:monospace;line-height:1.2em;word-wrap:break-word;box-shadow:0 0 6px rgba(0,0,0,.3)}.reveal code{font-family:monospace}.reveal pre code{padding:5px;overflow:auto;max-height:400px;word-wrap:normal}.reveal table th,.reveal table td{text-align:left;padding-right:.3em}.reveal table th{text-shadow:#fff 1px 1px 2px}.reveal sup{vertical-align:super}.reveal sub{vertical-align:sub}.reveal small{display:inline-block;font-size:.6em;line-height:1.2em;vertical-align:top}.reveal small *{vertical-align:top}.reveal .controls{display:none;position:fixed;width:110px;height:110px;z-index:30;right:10px;bottom:10px}.reveal .controls div{position:absolute;opacity:.05;width:0;height:0;border:12px solid transparent;-moz-transform:scale(.9999);-webkit-transition:all .2s ease;-moz-transition:all .2s ease;-ms-transition:all .2s ease;-o-transition:all .2s ease;transition:all .2s ease}.reveal .controls div.enabled{opacity:.7;cursor:pointer}.reveal .controls div.enabled:active{margin-top:1px}.reveal .controls div.navigate-left{top:42px;border-right-width:22px;border-right-color:#eee}.reveal .controls div.navigate-left.fragmented{opacity:.3}.reveal .controls div.navigate-right{left:74px;top:42px;border-left-width:22px;border-left-color:#eee}.reveal .controls div.navigate-right.fragmented{opacity:.3}.reveal .controls div.navigate-up{left:42px;border-bottom-width:22px;border-bottom-color:#eee}.reveal .controls div.navigate-up.fragmented{opacity:.3}.reveal .controls div.navigate-down{left:42px;top:74px;border-top-width:22px;border-top-color:#eee}.reveal .controls div.navigate-down.fragmented{opacity:.3}.reveal .progress{position:fixed;display:none;height:3px;width:100%;bottom:0;left:0;z-index:10}.reveal .progress:after{content:'';display:'block';position:absolute;height:20px;width:100%;top:-20px}.reveal .progress span{display:block;height:100%;width:0;-webkit-transition:width 800ms cubic-bezier(0.26,.86,.44,.985);-moz-transition:width 800ms cubic-bezier(0.26,.86,.44,.985);-ms-transition:width 800ms cubic-bezier(0.26,.86,.44,.985);-o-transition:width 800ms cubic-bezier(0.26,.86,.44,.985);transition:width 800ms cubic-bezier(0.26,.86,.44,.985)}.reveal .roll{display:inline-block;line-height:1.2;overflow:hidden;vertical-align:top;-webkit-perspective:400px;-moz-perspective:400px;-ms-perspective:400px;perspective:400px;-webkit-perspective-origin:50% 50%;-moz-perspective-origin:50% 50%;-ms-perspective-origin:50% 50%;perspective-origin:50% 50%}.reveal .roll:hover{background:0;text-shadow:none}.reveal .roll span{display:block;position:relative;padding:0 2px;pointer-events:none;-webkit-transition:all 400ms ease;-moz-transition:all 400ms ease;-ms-transition:all 400ms ease;transition:all 400ms ease;-webkit-transform-origin:50% 0;-moz-transform-origin:50% 0;-ms-transform-origin:50% 0;transform-origin:50% 0;-webkit-transform-style:preserve-3d;-moz-transform-style:preserve-3d;-ms-transform-style:preserve-3d;transform-style:preserve-3d;-webkit-backface-visibility:hidden;-moz-backface-visibility:hidden;backface-visibility:hidden}.reveal .roll:hover span{background:rgba(0,0,0,.5);-webkit-transform:translate3d(0px,0,-45px) rotateX(90deg);-moz-transform:translate3d(0px,0,-45px) rotateX(90deg);-ms-transform:translate3d(0px,0,-45px) rotateX(90deg);transform:translate3d(0px,0,-45px) rotateX(90deg)}.reveal .roll span:after{content:attr(data-title);display:block;position:absolute;left:0;top:0;padding:0 2px;-webkit-backface-visibility:hidden;-moz-backface-visibility:hidden;backface-visibility:hidden;-webkit-transform-origin:50% 0;-moz-transform-origin:50% 0;-ms-transform-origin:50% 0;transform-origin:50% 0;-webkit-transform:translate3d(0px,110%,0) rotateX(-90deg);-moz-transform:translate3d(0px,110%,0) rotateX(-90deg);-ms-transform:translate3d(0px,110%,0) rotateX(-90deg);transform:translate3d(0px,110%,0) rotateX(-90deg)}.reveal{position:relative;width:100%;height:100%;-ms-touch-action:none}.reveal .slides{position:absolute;width:100%;height:100%;left:50%;top:50%;overflow:visible;z-index:1;text-align:center;-webkit-transition:-webkit-perspective .4s ease;-moz-transition:-moz-perspective .4s ease;-ms-transition:-ms-perspective .4s ease;-o-transition:-o-perspective .4s ease;transition:perspective .4s ease;-webkit-perspective:600px;-moz-perspective:600px;-ms-perspective:600px;perspective:600px;-webkit-perspective-origin:0 -100px;-moz-perspective-origin:0 -100px;-ms-perspective-origin:0 -100px;perspective-origin:0 -100px}.reveal .slides>section,.reveal .slides>section>section{display:none;position:absolute;width:100%;padding:20px 0;z-index:10;line-height:1.2em;font-weight:400;-webkit-transform-style:preserve-3d;-moz-transform-style:preserve-3d;-ms-transform-style:preserve-3d;transform-style:preserve-3d;-webkit-transition:-webkit-transform-origin 800ms cubic-bezier(0.26,.86,.44,.985),-webkit-transform 800ms cubic-bezier(0.26,.86,.44,.985),visibility 800ms cubic-bezier(0.26,.86,.44,.985),opacity 800ms cubic-bezier(0.26,.86,.44,.985);-moz-transition:-moz-transform-origin 800ms cubic-bezier(0.26,.86,.44,.985),-moz-transform 800ms cubic-bezier(0.26,.86,.44,.985),visibility 800ms cubic-bezier(0.26,.86,.44,.985),opacity 800ms cubic-bezier(0.26,.86,.44,.985);-ms-transition:-ms-transform-origin 800ms cubic-bezier(0.26,.86,.44,.985),-ms-transform 800ms cubic-bezier(0.26,.86,.44,.985),visibility 800ms cubic-bezier(0.26,.86,.44,.985),opacity 800ms cubic-bezier(0.26,.86,.44,.985);-o-transition:-o-transform-origin 800ms cubic-bezier(0.26,.86,.44,.985),-o-transform 800ms cubic-bezier(0.26,.86,.44,.985),visibility 800ms cubic-bezier(0.26,.86,.44,.985),opacity 800ms cubic-bezier(0.26,.86,.44,.985);transition:transform-origin 800ms cubic-bezier(0.26,.86,.44,.985),transform 800ms cubic-bezier(0.26,.86,.44,.985),visibility 800ms cubic-bezier(0.26,.86,.44,.985),opacity 800ms cubic-bezier(0.26,.86,.44,.985)}.reveal[data-transition-speed=fast] .slides section{-webkit-transition-duration:400ms;-moz-transition-duration:400ms;-ms-transition-duration:400ms;transition-duration:400ms}.reveal[data-transition-speed=slow] .slides section{-webkit-transition-duration:1200ms;-moz-transition-duration:1200ms;-ms-transition-duration:1200ms;transition-duration:1200ms}.reveal .slides section[data-transition-speed=fast]{-webkit-transition-duration:400ms;-moz-transition-duration:400ms;-ms-transition-duration:400ms;transition-duration:400ms}.reveal .slides section[data-transition-speed=slow]{-webkit-transition-duration:1200ms;-moz-transition-duration:1200ms;-ms-transition-duration:1200ms;transition-duration:1200ms}.reveal .slides>section{left:-50%;top:-50%}.reveal .slides>section.stack{padding-top:0;padding-bottom:0}.reveal .slides>section.present,.reveal .slides>section>section.present{display:block;z-index:11;opacity:1}.reveal.center,.reveal.center .slides,.reveal.center .slides section{min-height:auto!important}.reveal .slides>section[data-transition=default].past,.reveal .slides>section.past{display:block;opacity:0;-webkit-transform:translate3d(-100%,0,0) rotateY(-90deg) translate3d(-100%,0,0);-moz-transform:translate3d(-100%,0,0) rotateY(-90deg) translate3d(-100%,0,0);-ms-transform:translate3d(-100%,0,0) rotateY(-90deg) translate3d(-100%,0,0);transform:translate3d(-100%,0,0) rotateY(-90deg) translate3d(-100%,0,0)}.reveal .slides>section[data-transition=default].future,.reveal .slides>section.future{display:block;opacity:0;-webkit-transform:translate3d(100%,0,0) rotateY(90deg) translate3d(100%,0,0);-moz-transform:translate3d(100%,0,0) rotateY(90deg) translate3d(100%,0,0);-ms-transform:translate3d(100%,0,0) rotateY(90deg) translate3d(100%,0,0);transform:translate3d(100%,0,0) rotateY(90deg) translate3d(100%,0,0)}.reveal .slides>section>section[data-transition=default].past,.reveal .slides>section>section.past{display:block;opacity:0;-webkit-transform:translate3d(0,-300px,0) rotateX(70deg) translate3d(0,-300px,0);-moz-transform:translate3d(0,-300px,0) rotateX(70deg) translate3d(0,-300px,0);-ms-transform:translate3d(0,-300px,0) rotateX(70deg) translate3d(0,-300px,0);transform:translate3d(0,-300px,0) rotateX(70deg) translate3d(0,-300px,0)}.reveal .slides>section>section[data-transition=default].future,.reveal .slides>section>section.future{display:block;opacity:0;-webkit-transform:translate3d(0,300px,0) rotateX(-70deg) translate3d(0,300px,0);-moz-transform:translate3d(0,300px,0) rotateX(-70deg) translate3d(0,300px,0);-ms-transform:translate3d(0,300px,0) rotateX(-70deg) translate3d(0,300px,0);transform:translate3d(0,300px,0) rotateX(-70deg) translate3d(0,300px,0)}.reveal .slides>section[data-transition=concave].past,.reveal.concave .slides>section.past{-webkit-transform:translate3d(-100%,0,0) rotateY(90deg) translate3d(-100%,0,0);-moz-transform:translate3d(-100%,0,0) rotateY(90deg) translate3d(-100%,0,0);-ms-transform:translate3d(-100%,0,0) rotateY(90deg) translate3d(-100%,0,0);transform:translate3d(-100%,0,0) rotateY(90deg) translate3d(-100%,0,0)}.reveal .slides>section[data-transition=concave].future,.reveal.concave .slides>section.future{-webkit-transform:translate3d(100%,0,0) rotateY(-90deg) translate3d(100%,0,0);-moz-transform:translate3d(100%,0,0) rotateY(-90deg) translate3d(100%,0,0);-ms-transform:translate3d(100%,0,0) rotateY(-90deg) translate3d(100%,0,0);transform:translate3d(100%,0,0) rotateY(-90deg) translate3d(100%,0,0)}.reveal .slides>section>section[data-transition=concave].past,.reveal.concave .slides>section>section.past{-webkit-transform:translate3d(0,-80%,0) rotateX(-70deg) translate3d(0,-80%,0);-moz-transform:translate3d(0,-80%,0) rotateX(-70deg) translate3d(0,-80%,0);-ms-transform:translate3d(0,-80%,0) rotateX(-70deg) translate3d(0,-80%,0);transform:translate3d(0,-80%,0) rotateX(-70deg) translate3d(0,-80%,0)}.reveal .slides>section>section[data-transition=concave].future,.reveal.concave .slides>section>section.future{-webkit-transform:translate3d(0,80%,0) rotateX(70deg) translate3d(0,80%,0);-moz-transform:translate3d(0,80%,0) rotateX(70deg) translate3d(0,80%,0);-ms-transform:translate3d(0,80%,0) rotateX(70deg) translate3d(0,80%,0);transform:translate3d(0,80%,0) rotateX(70deg) translate3d(0,80%,0)}.reveal .slides>section[data-transition=zoom].past,.reveal.zoom .slides>section.past{opacity:0;visibility:hidden;-webkit-transform:scale(16);-moz-transform:scale(16);-ms-transform:scale(16);-o-transform:scale(16);transform:scale(16)}.reveal .slides>section[data-transition=zoom].future,.reveal.zoom .slides>section.future{opacity:0;visibility:hidden;-webkit-transform:scale(0.2);-moz-transform:scale(0.2);-ms-transform:scale(0.2);-o-transform:scale(0.2);transform:scale(0.2)}.reveal .slides>section>section[data-transition=zoom].past,.reveal.zoom .slides>section>section.past{-webkit-transform:translate(0,-150%);-moz-transform:translate(0,-150%);-ms-transform:translate(0,-150%);-o-transform:translate(0,-150%);transform:translate(0,-150%)}.reveal .slides>section>section[data-transition=zoom].future,.reveal.zoom .slides>section>section.future{-webkit-transform:translate(0,150%);-moz-transform:translate(0,150%);-ms-transform:translate(0,150%);-o-transform:translate(0,150%);transform:translate(0,150%)}.reveal.linear section{-webkit-backface-visibility:hidden;-moz-backface-visibility:hidden;-ms-backface-visibility:hidden;backface-visibility:hidden}.reveal .slides>section[data-transition=linear].past,.reveal.linear .slides>section.past{-webkit-transform:translate(-150%,0);-moz-transform:translate(-150%,0);-ms-transform:translate(-150%,0);-o-transform:translate(-150%,0);transform:translate(-150%,0)}.reveal .slides>section[data-transition=linear].future,.reveal.linear .slides>section.future{-webkit-transform:translate(150%,0);-moz-transform:translate(150%,0);-ms-transform:translate(150%,0);-o-transform:translate(150%,0);transform:translate(150%,0)}.reveal .slides>section>section[data-transition=linear].past,.reveal.linear .slides>section>section.past{-webkit-transform:translate(0,-150%);-moz-transform:translate(0,-150%);-ms-transform:translate(0,-150%);-o-transform:translate(0,-150%);transform:translate(0,-150%)}.reveal .slides>section>section[data-transition=linear].future,.reveal.linear .slides>section>section.future{-webkit-transform:translate(0,150%);-moz-transform:translate(0,150%);-ms-transform:translate(0,150%);-o-transform:translate(0,150%);transform:translate(0,150%)}.reveal.cube .slides{-webkit-perspective:1300px;-moz-perspective:1300px;-ms-perspective:1300px;perspective:1300px}.reveal.cube .slides section{padding:30px;min-height:700px;-webkit-backface-visibility:hidden;-moz-backface-visibility:hidden;-ms-backface-visibility:hidden;backface-visibility:hidden;-webkit-box-sizing:border-box;-moz-box-sizing:border-box;box-sizing:border-box}.reveal.center.cube .slides section{min-height:auto}.reveal.cube .slides section:not(.stack):before{content:'';position:absolute;display:block;width:100%;height:100%;left:0;top:0;background:rgba(0,0,0,.1);border-radius:4px;-webkit-transform:translateZ(-20px);-moz-transform:translateZ(-20px);-ms-transform:translateZ(-20px);-o-transform:translateZ(-20px);transform:translateZ(-20px)}.reveal.cube .slides section:not(.stack):after{content:'';position:absolute;display:block;width:90%;height:30px;left:5%;bottom:0;background:0;z-index:1;border-radius:4px;box-shadow:0 95px 25px rgba(0,0,0,.2);-webkit-transform:translateZ(-90px) rotateX(65deg);-moz-transform:translateZ(-90px) rotateX(65deg);-ms-transform:translateZ(-90px) rotateX(65deg);-o-transform:translateZ(-90px) rotateX(65deg);transform:translateZ(-90px) rotateX(65deg)}.reveal.cube .slides>section.stack{padding:0;background:0}.reveal.cube .slides>section.past{-webkit-transform-origin:100% 0;-moz-transform-origin:100% 0;-ms-transform-origin:100% 0;transform-origin:100% 0;-webkit-transform:translate3d(-100%,0,0) rotateY(-90deg);-moz-transform:translate3d(-100%,0,0) rotateY(-90deg);-ms-transform:translate3d(-100%,0,0) rotateY(-90deg);transform:translate3d(-100%,0,0) rotateY(-90deg)}.reveal.cube .slides>section.future{-webkit-transform-origin:0 0;-moz-transform-origin:0 0;-ms-transform-origin:0 0;transform-origin:0 0;-webkit-transform:translate3d(100%,0,0) rotateY(90deg);-moz-transform:translate3d(100%,0,0) rotateY(90deg);-ms-transform:translate3d(100%,0,0) rotateY(90deg);transform:translate3d(100%,0,0) rotateY(90deg)}.reveal.cube .slides>section>section.past{-webkit-transform-origin:0 100%;-moz-transform-origin:0 100%;-ms-transform-origin:0 100%;transform-origin:0 100%;-webkit-transform:translate3d(0,-100%,0) rotateX(90deg);-moz-transform:translate3d(0,-100%,0) rotateX(90deg);-ms-transform:translate3d(0,-100%,0) rotateX(90deg);transform:translate3d(0,-100%,0) rotateX(90deg)}.reveal.cube .slides>section>section.future{-webkit-transform-origin:0 0;-moz-transform-origin:0 0;-ms-transform-origin:0 0;transform-origin:0 0;-webkit-transform:translate3d(0,100%,0) rotateX(-90deg);-moz-transform:translate3d(0,100%,0) rotateX(-90deg);-ms-transform:translate3d(0,100%,0) rotateX(-90deg);transform:translate3d(0,100%,0) rotateX(-90deg)}.reveal.page .slides{-webkit-perspective-origin:0 50%;-moz-perspective-origin:0 50%;-ms-perspective-origin:0 50%;perspective-origin:0 50%;-webkit-perspective:3000px;-moz-perspective:3000px;-ms-perspective:3000px;perspective:3000px}.reveal.page .slides section{padding:30px;min-height:700px;-webkit-box-sizing:border-box;-moz-box-sizing:border-box;box-sizing:border-box}.reveal.page .slides section.past{z-index:12}.reveal.page .slides section:not(.stack):before{content:'';position:absolute;display:block;width:100%;height:100%;left:0;top:0;background:rgba(0,0,0,.1);-webkit-transform:translateZ(-20px);-moz-transform:translateZ(-20px);-ms-transform:translateZ(-20px);-o-transform:translateZ(-20px);transform:translateZ(-20px)}.reveal.page .slides section:not(.stack):after{content:'';position:absolute;display:block;width:90%;height:30px;left:5%;bottom:0;background:0;z-index:1;border-radius:4px;box-shadow:0 95px 25px rgba(0,0,0,.2);-webkit-transform:translateZ(-90px) rotateX(65deg)}.reveal.page .slides>section.stack{padding:0;background:0}.reveal.page .slides>section.past{-webkit-transform-origin:0 0;-moz-transform-origin:0 0;-ms-transform-origin:0 0;transform-origin:0 0;-webkit-transform:translate3d(-40%,0,0) rotateY(-80deg);-moz-transform:translate3d(-40%,0,0) rotateY(-80deg);-ms-transform:translate3d(-40%,0,0) rotateY(-80deg);transform:translate3d(-40%,0,0) rotateY(-80deg)}.reveal.page .slides>section.future{-webkit-transform-origin:100% 0;-moz-transform-origin:100% 0;-ms-transform-origin:100% 0;transform-origin:100% 0;-webkit-transform:translate3d(0,0,0);-moz-transform:translate3d(0,0,0);-ms-transform:translate3d(0,0,0);transform:translate3d(0,0,0)}.reveal.page .slides>section>section.past{-webkit-transform-origin:0 0;-moz-transform-origin:0 0;-ms-transform-origin:0 0;transform-origin:0 0;-webkit-transform:translate3d(0,-40%,0) rotateX(80deg);-moz-transform:translate3d(0,-40%,0) rotateX(80deg);-ms-transform:translate3d(0,-40%,0) rotateX(80deg);transform:translate3d(0,-40%,0) rotateX(80deg)}.reveal.page .slides>section>section.future{-webkit-transform-origin:0 100%;-moz-transform-origin:0 100%;-ms-transform-origin:0 100%;transform-origin:0 100%;-webkit-transform:translate3d(0,0,0);-moz-transform:translate3d(0,0,0);-ms-transform:translate3d(0,0,0);transform:translate3d(0,0,0)}.reveal .slides section[data-transition=fade],.reveal.fade .slides section,.reveal.fade .slides>section>section{-webkit-transform:none;-moz-transform:none;-ms-transform:none;-o-transform:none;transform:none;-webkit-transition:opacity .5s;-moz-transition:opacity .5s;-ms-transition:opacity .5s;-o-transition:opacity .5s;transition:opacity .5s}.reveal.fade.overview .slides section,.reveal.fade.overview .slides>section>section,.reveal.fade.exit-overview .slides section,.reveal.fade.exit-overview .slides>section>section{-webkit-transition:none;-moz-transition:none;-ms-transition:none;-o-transition:none;transition:none}.reveal .slides section[data-transition=none],.reveal.none .slides section{-webkit-transform:none;-moz-transform:none;-ms-transform:none;-o-transform:none;transform:none;-webkit-transition:none;-moz-transition:none;-ms-transition:none;-o-transition:none;transition:none}.reveal.overview .slides{-webkit-perspective-origin:0 0;-moz-perspective-origin:0 0;-ms-perspective-origin:0 0;perspective-origin:0 0;-webkit-perspective:700px;-moz-perspective:700px;-ms-perspective:700px;perspective:700px}.reveal.overview .slides section{height:600px;overflow:hidden;opacity:1!important;visibility:visible!important;cursor:pointer;background:rgba(0,0,0,.1)}.reveal.overview .slides section .fragment{opacity:1}.reveal.overview .slides section:after,.reveal.overview .slides section:before{display:none!important}.reveal.overview .slides section>section{opacity:1;cursor:pointer}.reveal.overview .slides section:hover{background:rgba(0,0,0,.3)}.reveal.overview .slides section.present{background:rgba(0,0,0,.3)}.reveal.overview .slides>section.stack{padding:0;background:0;overflow:visible}.reveal .pause-overlay{position:absolute;top:0;left:0;width:100%;height:100%;background:#000;visibility:hidden;opacity:0;z-index:100;-webkit-transition:all 1s ease;-moz-transition:all 1s ease;-ms-transition:all 1s ease;-o-transition:all 1s ease;transition:all 1s ease}.reveal.paused .pause-overlay{visibility:visible;opacity:1}.no-transforms{overflow-y:auto}.no-transforms .reveal .slides{position:relative;width:80%;height:auto!important;top:0;left:50%;margin:0;text-align:center}.no-transforms .reveal .controls,.no-transforms .reveal .progress{display:none!important}.no-transforms .reveal .slides section{display:block!important;opacity:1!important;position:relative!important;height:auto;min-height:auto;top:0;left:-50%;margin:70px 0;-webkit-transform:none;-moz-transform:none;-ms-transform:none;-o-transform:none;transform:none}.no-transforms .reveal .slides section section{left:0}.reveal .no-transition,.reveal .no-transition *{-webkit-transition:none!important;-moz-transition:none!important;-ms-transition:none!important;-o-transition:none!important;transition:none!important}.reveal .state-background{position:absolute;width:100%;height:100%;background:rgba(0,0,0,0);-webkit-transition:background 800ms ease;-moz-transition:background 800ms ease;-ms-transition:background 800ms ease;-o-transition:background 800ms ease;transition:background 800ms ease}.alert .reveal .state-background{background:rgba(200,50,30,.6)}.soothe .reveal .state-background{background:rgba(50,200,90,.4)}.blackout .reveal .state-background{background:rgba(0,0,0,.6)}.whiteout .reveal .state-background{background:rgba(255,255,255,.6)}.cobalt .reveal .state-background{background:rgba(22,152,213,.6)}.mint .reveal .state-background{background:rgba(22,213,75,.6)}.submerge .reveal .state-background{background:rgba(12,25,77,.6)}.lila .reveal .state-background{background:rgba(180,50,140,.6)}.sunset .reveal .state-background{background:rgba(255,122,0,.6)}.reveal>.backgrounds{position:absolute;width:100%;height:100%}.reveal .slide-background{position:absolute;width:100%;height:100%;opacity:0;visibility:hidden;background-color:rgba(0,0,0,0);background-position:50% 50%;background-repeat:no-repeat;background-size:cover;-webkit-transition:all 600ms cubic-bezier(0.26,.86,.44,.985);-moz-transition:all 600ms cubic-bezier(0.26,.86,.44,.985);-ms-transition:all 600ms cubic-bezier(0.26,.86,.44,.985);-o-transition:all 600ms cubic-bezier(0.26,.86,.44,.985);transition:all 600ms cubic-bezier(0.26,.86,.44,.985)}.reveal .slide-background.present{opacity:1;visibility:visible}.print-pdf .reveal .slide-background{opacity:1!important;visibility:visible!important}.reveal[data-background-transition=slide]>.backgrounds .slide-background,.reveal>.backgrounds .slide-background[data-background-transition=slide]{opacity:1;-webkit-backface-visibility:hidden;-moz-backface-visibility:hidden;-ms-backface-visibility:hidden;backface-visibility:hidden;-webkit-transition-duration:800ms;-moz-transition-duration:800ms;-ms-transition-duration:800ms;-o-transition-duration:800ms;transition-duration:800ms}.reveal[data-background-transition=slide]>.backgrounds .slide-background.past,.reveal>.backgrounds .slide-background.past[data-background-transition=slide]{-webkit-transform:translate(-100%,0);-moz-transform:translate(-100%,0);-ms-transform:translate(-100%,0);-o-transform:translate(-100%,0);transform:translate(-100%,0)}.reveal[data-background-transition=slide]>.backgrounds .slide-background.future,.reveal>.backgrounds .slide-background.future[data-background-transition=slide]{-webkit-transform:translate(100%,0);-moz-transform:translate(100%,0);-ms-transform:translate(100%,0);-o-transform:translate(100%,0);transform:translate(100%,0)}.reveal[data-background-transition=slide]>.backgrounds .slide-background>.slide-background.past,.reveal>.backgrounds .slide-background>.slide-background.past[data-background-transition=slide]{-webkit-transform:translate(0,-100%);-moz-transform:translate(0,-100%);-ms-transform:translate(0,-100%);-o-transform:translate(0,-100%);transform:translate(0,-100%)}.reveal[data-background-transition=slide]>.backgrounds .slide-background>.slide-background.future,.reveal>.backgrounds .slide-background>.slide-background.future[data-background-transition=slide]{-webkit-transform:translate(0,100%);-moz-transform:translate(0,100%);-ms-transform:translate(0,100%);-o-transform:translate(0,100%);transform:translate(0,100%)}.reveal[data-transition-speed=fast]>.backgrounds .slide-background{-webkit-transition-duration:400ms;-moz-transition-duration:400ms;-ms-transition-duration:400ms;transition-duration:400ms}.reveal[data-transition-speed=slow]>.backgrounds .slide-background{-webkit-transition-duration:1200ms;-moz-transition-duration:1200ms;-ms-transition-duration:1200ms;transition-duration:1200ms}.reveal.rtl .slides,.reveal.rtl .slides h1,.reveal.rtl .slides h2,.reveal.rtl .slides h3,.reveal.rtl .slides h4,.reveal.rtl .slides h5,.reveal.rtl .slides h6{direction:rtl;font-family:sans-serif}.reveal.rtl pre,.reveal.rtl code{direction:ltr}.reveal.rtl ol,.reveal.rtl ul{text-align:right}.reveal.rtl .progress span{float:right}.reveal .preview-link-overlay{position:absolute;top:0;left:0;width:100%;height:100%;z-index:1000;background:rgba(0,0,0,.9);opacity:0;visibility:hidden;-webkit-transition:all .3s ease;-moz-transition:all .3s ease;-ms-transition:all .3s ease;transition:all .3s ease}.reveal .preview-link-overlay.visible{opacity:1;visibility:visible}.reveal .preview-link-overlay .spinner{position:absolute;display:block;top:50%;left:50%;width:32px;height:32px;margin:-16px 0 0 -16px;z-index:10;background-image:url(data:image/gif;base64,R0lGODlhIAAgAPMAAJmZmf%2F%2F%2F6%2Bvr8nJybW1tcDAwOjo6Nvb26ioqKOjo7Ozs%2FLy8vz8%2FAAAAAAAAAAAACH%2FC05FVFNDQVBFMi4wAwEAAAAh%2FhpDcmVhdGVkIHdpdGggYWpheGxvYWQuaW5mbwAh%2BQQJCgAAACwAAAAAIAAgAAAE5xDISWlhperN52JLhSSdRgwVo1ICQZRUsiwHpTJT4iowNS8vyW2icCF6k8HMMBkCEDskxTBDAZwuAkkqIfxIQyhBQBFvAQSDITM5VDW6XNE4KagNh6Bgwe60smQUB3d4Rz1ZBApnFASDd0hihh12BkE9kjAJVlycXIg7CQIFA6SlnJ87paqbSKiKoqusnbMdmDC2tXQlkUhziYtyWTxIfy6BE8WJt5YJvpJivxNaGmLHT0VnOgSYf0dZXS7APdpB309RnHOG5gDqXGLDaC457D1zZ%2FV%2FnmOM82XiHRLYKhKP1oZmADdEAAAh%2BQQJCgAAACwAAAAAIAAgAAAE6hDISWlZpOrNp1lGNRSdRpDUolIGw5RUYhhHukqFu8DsrEyqnWThGvAmhVlteBvojpTDDBUEIFwMFBRAmBkSgOrBFZogCASwBDEY%2FCZSg7GSE0gSCjQBMVG023xWBhklAnoEdhQEfyNqMIcKjhRsjEdnezB%2BA4k8gTwJhFuiW4dokXiloUepBAp5qaKpp6%2BHo7aWW54wl7obvEe0kRuoplCGepwSx2jJvqHEmGt6whJpGpfJCHmOoNHKaHx61WiSR92E4lbFoq%2BB6QDtuetcaBPnW6%2BO7wDHpIiK9SaVK5GgV543tzjgGcghAgAh%2BQQJCgAAACwAAAAAIAAgAAAE7hDISSkxpOrN5zFHNWRdhSiVoVLHspRUMoyUakyEe8PTPCATW9A14E0UvuAKMNAZKYUZCiBMuBakSQKG8G2FzUWox2AUtAQFcBKlVQoLgQReZhQlCIJesQXI5B0CBnUMOxMCenoCfTCEWBsJColTMANldx15BGs8B5wlCZ9Po6OJkwmRpnqkqnuSrayqfKmqpLajoiW5HJq7FL1Gr2mMMcKUMIiJgIemy7xZtJsTmsM4xHiKv5KMCXqfyUCJEonXPN2rAOIAmsfB3uPoAK%2B%2BG%2Bw48edZPK%2BM6hLJpQg484enXIdQFSS1u6UhksENEQAAIfkECQoAAAAsAAAAACAAIAAABOcQyEmpGKLqzWcZRVUQnZYg1aBSh2GUVEIQ2aQOE%2BG%2BcD4ntpWkZQj1JIiZIogDFFyHI0UxQwFugMSOFIPJftfVAEoZLBbcLEFhlQiqGp1Vd140AUklUN3eCA51C1EWMzMCezCBBmkxVIVHBWd3HHl9JQOIJSdSnJ0TDKChCwUJjoWMPaGqDKannasMo6WnM562R5YluZRwur0wpgqZE7NKUm%2BFNRPIhjBJxKZteWuIBMN4zRMIVIhffcgojwCF117i4nlLnY5ztRLsnOk%2BaV%2BoJY7V7m76PdkS4trKcdg0Zc0tTcKkRAAAIfkECQoAAAAsAAAAACAAIAAABO4QyEkpKqjqzScpRaVkXZWQEximw1BSCUEIlDohrft6cpKCk5xid5MNJTaAIkekKGQkWyKHkvhKsR7ARmitkAYDYRIbUQRQjWBwJRzChi9CRlBcY1UN4g0%2FVNB0AlcvcAYHRyZPdEQFYV8ccwR5HWxEJ02YmRMLnJ1xCYp0Y5idpQuhopmmC2KgojKasUQDk5BNAwwMOh2RtRq5uQuPZKGIJQIGwAwGf6I0JXMpC8C7kXWDBINFMxS4DKMAWVWAGYsAdNqW5uaRxkSKJOZKaU3tPOBZ4DuK2LATgJhkPJMgTwKCdFjyPHEnKxFCDhEAACH5BAkKAAAALAAAAAAgACAAAATzEMhJaVKp6s2nIkolIJ2WkBShpkVRWqqQrhLSEu9MZJKK9y1ZrqYK9WiClmvoUaF8gIQSNeF1Er4MNFn4SRSDARWroAIETg1iVwuHjYB1kYc1mwruwXKC9gmsJXliGxc%2BXiUCby9ydh1sOSdMkpMTBpaXBzsfhoc5l58Gm5yToAaZhaOUqjkDgCWNHAULCwOLaTmzswadEqggQwgHuQsHIoZCHQMMQgQGubVEcxOPFAcMDAYUA85eWARmfSRQCdcMe0zeP1AAygwLlJtPNAAL19DARdPzBOWSm1brJBi45soRAWQAAkrQIykShQ9wVhHCwCQCACH5BAkKAAAALAAAAAAgACAAAATrEMhJaVKp6s2nIkqFZF2VIBWhUsJaTokqUCoBq%2BE71SRQeyqUToLA7VxF0JDyIQh%2FMVVPMt1ECZlfcjZJ9mIKoaTl1MRIl5o4CUKXOwmyrCInCKqcWtvadL2SYhyASyNDJ0uIiRMDjI0Fd30%2FiI2UA5GSS5UDj2l6NoqgOgN4gksEBgYFf0FDqKgHnyZ9OX8HrgYHdHpcHQULXAS2qKpENRg7eAMLC7kTBaixUYFkKAzWAAnLC7FLVxLWDBLKCwaKTULgEwbLA4hJtOkSBNqITT3xEgfLpBtzE%2FjiuL04RGEBgwWhShRgQExHBAAh%2BQQJCgAAACwAAAAAIAAgAAAE7xDISWlSqerNpyJKhWRdlSAVoVLCWk6JKlAqAavhO9UkUHsqlE6CwO1cRdCQ8iEIfzFVTzLdRAmZX3I2SfZiCqGk5dTESJeaOAlClzsJsqwiJwiqnFrb2nS9kmIcgEsjQydLiIlHehhpejaIjzh9eomSjZR%2BipslWIRLAgMDOR2DOqKogTB9pCUJBagDBXR6XB0EBkIIsaRsGGMMAxoDBgYHTKJiUYEGDAzHC9EACcUGkIgFzgwZ0QsSBcXHiQvOwgDdEwfFs0sDzt4S6BK4xYjkDOzn0unFeBzOBijIm1Dgmg5YFQwsCMjp1oJ8LyIAACH5BAkKAAAALAAAAAAgACAAAATwEMhJaVKp6s2nIkqFZF2VIBWhUsJaTokqUCoBq%2BE71SRQeyqUToLA7VxF0JDyIQh%2FMVVPMt1ECZlfcjZJ9mIKoaTl1MRIl5o4CUKXOwmyrCInCKqcWtvadL2SYhyASyNDJ0uIiUd6GGl6NoiPOH16iZKNlH6KmyWFOggHhEEvAwwMA0N9GBsEC6amhnVcEwavDAazGwIDaH1ipaYLBUTCGgQDA8NdHz0FpqgTBwsLqAbWAAnIA4FWKdMLGdYGEgraigbT0OITBcg5QwPT4xLrROZL6AuQAPUS7bxLpoWidY0JtxLHKhwwMJBTHgPKdEQAACH5BAkKAAAALAAAAAAgACAAAATrEMhJaVKp6s2nIkqFZF2VIBWhUsJaTokqUCoBq%2BE71SRQeyqUToLA7VxF0JDyIQh%2FMVVPMt1ECZlfcjZJ9mIKoaTl1MRIl5o4CUKXOwmyrCInCKqcWtvadL2SYhyASyNDJ0uIiUd6GAULDJCRiXo1CpGXDJOUjY%2BYip9DhToJA4RBLwMLCwVDfRgbBAaqqoZ1XBMHswsHtxtFaH1iqaoGNgAIxRpbFAgfPQSqpbgGBqUD1wBXeCYp1AYZ19JJOYgH1KwA4UBvQwXUBxPqVD9L3sbp2BNk2xvvFPJd%2BMFCN6HAAIKgNggY0KtEBAAh%2BQQJCgAAACwAAAAAIAAgAAAE6BDISWlSqerNpyJKhWRdlSAVoVLCWk6JKlAqAavhO9UkUHsqlE6CwO1cRdCQ8iEIfzFVTzLdRAmZX3I2SfYIDMaAFdTESJeaEDAIMxYFqrOUaNW4E4ObYcCXaiBVEgULe0NJaxxtYksjh2NLkZISgDgJhHthkpU4mW6blRiYmZOlh4JWkDqILwUGBnE6TYEbCgevr0N1gH4At7gHiRpFaLNrrq8HNgAJA70AWxQIH1%2BvsYMDAzZQPC9VCNkDWUhGkuE5PxJNwiUK4UfLzOlD4WvzAHaoG9nxPi5d%2BjYUqfAhhykOFwJWiAAAIfkECQoAAAAsAAAAACAAIAAABPAQyElpUqnqzaciSoVkXVUMFaFSwlpOCcMYlErAavhOMnNLNo8KsZsMZItJEIDIFSkLGQoQTNhIsFehRww2CQLKF0tYGKYSg%2BygsZIuNqJksKgbfgIGepNo2cIUB3V1B3IvNiBYNQaDSTtfhhx0CwVPI0UJe0%2Bbm4g5VgcGoqOcnjmjqDSdnhgEoamcsZuXO1aWQy8KAwOAuTYYGwi7w5h%2BKr0SJ8MFihpNbx%2B4Erq7BYBuzsdiH1jCAzoSfl0rVirNbRXlBBlLX%2BBP0XJLAPGzTkAuAOqb0WT5AH7OcdCm5B8TgRwSRKIHQtaLCwg1RAAAOwAAAAAAAAAAAA%3D%3D);visibility:visible;opacity:.6;-webkit-transition:all .3s ease;-moz-transition:all .3s ease;-ms-transition:all .3s ease;transition:all .3s ease}.reveal .preview-link-overlay header{position:absolute;left:0;top:0;width:100%;height:40px;z-index:2;border-bottom:1px solid #222}.reveal .preview-link-overlay header a{display:inline-block;width:40px;height:40px;padding:0 10px;float:right;opacity:.6;box-sizing:border-box}.reveal .preview-link-overlay header a:hover{opacity:1}.reveal .preview-link-overlay header a .icon{display:inline-block;width:20px;height:20px;background-position:50% 50%;background-size:100%;background-repeat:no-repeat}.reveal .preview-link-overlay header a.close .icon{background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABkklEQVRYR8WX4VHDMAxG6wnoJrABZQPYBCaBTWAD2g1gE5gg6OOsXuxIlr40d81dfrSJ9V4c2VLK7spHuTJ/5wpM07QXuXc5X0opX2tEJcadjHuV80li/FgxTIEK/5QBCICBD6xEhSMGHgQPgBgLiYVAB1dpSqKDawxTohFw4JSEA3clzgIBPCURwE2JucBR7rhPJJv5OpJwDX+SfDjgx1wACQeJG1aChP9K/IMmdZ8DtESV1WyP3Bt4MwM6sj4NMxMYiqUWHQu4KYA/SYkIjOsm3BXYWMKFDwU2khjCQ4ELJUJ4SmClRArOCmSXGuKma0fYD5CbzHxFpCSGAhfAVSSUGDUk2BWZaff2g6GE15BsBQ9nwmpIGDiyHQddwNTMKkbZaf9fajXQca1EX44puJZUsnY0ObGmITE3GVLCbEhQUjGVt146j6oasWN+49Vph2w1pZ5EansNZqKBm1txbU57iRRcZ86RWMDdWtBJUHBHwoQPi1GV+JCbntmvok7iTX4/Up9mgyTc/FJYDTcndgH/AA5A/CHsyEkVAAAAAElFTkSuQmCC)}.reveal .preview-link-overlay header a.external .icon{background-image:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAcElEQVRYR+2WSQoAIQwEzf8f7XiOMkUQxUPlGkM3hVmiQfQR9GYnH1SsAQlI4DiBqkCMoNb9y2e90IAEJPAcgdznU9+engMaeJ7Azh5Y1U67gAho4DqBqmB1buAf0MB1AlVBek83ZPkmJMGc1wAR+AAqod/B97TRpQAAAABJRU5ErkJggg==)}.reveal .preview-link-overlay .viewport{position:absolute;top:40px;right:0;bottom:0;left:0}.reveal .preview-link-overlay .viewport iframe{width:100%;height:100%;max-width:100%;max-height:100%;border:0;opacity:0;visibility:hidden;-webkit-transition:all .3s ease;-moz-transition:all .3s ease;-ms-transition:all .3s ease;transition:all .3s ease}.reveal .preview-link-overlay.loaded .viewport iframe{opacity:1;visibility:visible}.reveal .preview-link-overlay.loaded .spinner{opacity:0;visibility:hidden;-webkit-transform:scale(0.2);-moz-transform:scale(0.2);-ms-transform:scale(0.2);transform:scale(0.2)}.reveal aside.notes{display:none}.zoomed .reveal *,.zoomed .reveal :before,.zoomed .reveal :after{-webkit-transform:none!important;-moz-transform:none!important;-ms-transform:none!important;transform:none!important;-webkit-backface-visibility:visible!important;-moz-backface-visibility:visible!important;-ms-backface-visibility:visible!important;backface-visibility:visible!important}.zoomed .reveal .progress,.zoomed .reveal .controls{opacity:0}.zoomed .reveal .roll span{background:0}.zoomed .reveal .roll span:after{visibility:hidden}
posts/conferences/bioindocker_day1.org:available from local companies. For health records, use [[https://github.com/OpenClinica/OpenClinica][OpenClinica]]. It's an
posts/conferences/scipy2013_day1.org:Deep learning requires a lot of labeled data and GPU enabled code to
posts/conferences/mtsinai_workshop.org:To parallelize better, have been focusing on 3 areas: AVX, GPU, FPGA. Started

```