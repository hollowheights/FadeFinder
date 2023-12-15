Status - current limitations - updated 15.12.2023
* hvis databasen mangler data for dage, vil de stadig logges som værende processed
* kan adde et step hvor jeg loader data og tjekker hvilke dage mangler i db'en - ret simpelt?
* scriptet skal køre til ende fordi signal-loggingen først stores permanent ved afslutning, modsat af logging af hvilke dage er processeret
 
Changes in version 4
-preprocessing properly included
	Now starts out by checking for dates already processed - query only relevant data
-D2 implemented 

The plan for version 4 was:
* expand to more than one type of trade setup
* make it run much faster than version 3
* make the code structure cleaner



