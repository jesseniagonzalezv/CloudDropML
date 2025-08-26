

procedure processFile_level2(cloudfile : string, geolocfile : string, split : string, output : string)
begin

  ; Open the files
  f1 = addfile(cloudfile, "r")
  g1 = addfile(geolocfile, "r")
  
    ref1s = f1->Cloud_Effective_Radius 
    cod1s = f1->Cloud_Optical_Thickness 
    lwp1s = f1->Cloud_Water_Path 
    ctt1s = f1->cloud_top_temperature_1km
    ctp1s = f1->cloud_top_pressure_1km
    cmf1s = f1->Cloud_Multi_Layer_Flag 
    cldph1s = f1->Cloud_Phase_Infrared_1km
    ;printVarSummary(cldph1s)

;  Apply scale and offset and convert to double
   ref1 =  ref1s@scale_factor*1.d * (ref1s - ref1s@add_offset)
   cod1 =  cod1s@scale_factor*1.d * (cod1s - cod1s@add_offset) 
   lwp1 =  lwp1s@scale_factor*1.d * (lwp1s - lwp1s@add_offset) 
   ctt1 =  ctt1s@scale_factor*1.d * (ctt1s - ctt1s@add_offset)
   ctp1 =  ctp1s@scale_factor*1.d * (ctp1s - ctp1s@add_offset) 
   cmf1 =  cmf1s@scale_factor*1.d * (cmf1s - cmf1s@add_offset) 
   cldph1 =  cldph1s@scale_factor*1.d * (cldph1s - cldph1s@add_offset)

    ;printMinMax(ref1, False)

   lat2d1    = g1->Latitude
   lon2d1    = g1->Longitude

   lon2d1!0 = "lat"
   lon2d1!1 = "lon"
   lat2d1!0 = "lat"
   lat2d1!1 = "lon"

   Longitude = lon2d1(0,:)
   Latitude = lat2d1(:,0)
   Latitude@units= "degrees_north"
   Longitude@units= "degrees_east"
   Latitude!0    = "Latitude"
   Longitude!0    = "Longitude"
   Latitude&Latitude  =  Latitude
   Longitude&Longitude  =  Longitude
     
  copy_VarCoords(lat2d1,ref1)
  copy_VarCoords(lat2d1,cod1)
  copy_VarCoords(lat2d1,lwp1)
  copy_VarCoords(lat2d1,ctt1)
  copy_VarCoords(lat2d1,ctp1)
  copy_VarCoords(lat2d1,cmf1)
  copy_VarCoords(lat2d1,cldph1)

;printVarSummary(lwp1)

;;;;;;;;;;;;;;;;;;;write netcdf;;;;;;;;;;;;;;;
  cdfName1 =output+split(0)+".nc"
  print("---------------------------------------- Saving file: " + cdfName1)
  system("/bin/rm -f "+cdfName1)         ; remove any pre-existing file
  
  ncdf1     = addfile(cdfName1 ,"c")      ; open output netCDF file;

  ncdf1->ref = ref1
  ncdf1->cod = cod1
  ncdf1->lwp = lwp1
  ncdf1->ctt = ctt1
  ncdf1->ctp = ctp1
  ncdf1->cmf = cmf1
  ncdf1->cldph = cldph1

  ncdf1->lat  = lat2d1
  ncdf1->lon = lon2d1   
  ncdf1->lat@units = "degrees_north"
  ncdf1->lon@units = "degrees_east"

  ncdf1->Latitude  = Latitude
  ncdf1->Longitude = Longitude   


  delete(ref1)
  delete(ref1s)
  delete(cod1)
  delete(cod1s)
  delete(lwp1)
  delete(lwp1s)
  delete(ctt1)
  delete(ctt1s)
  delete(ctp1)
  delete(ctp1s)
  delete(cmf1)
  delete(cmf1s)
  delete(cldph1)
  delete(Latitude)
  delete(Longitude)  
  delete(lat2d1)
  delete(lon2d1)
end



begin
  ; Read the list of matching cloud and geoloc filenames
    cloud_files = asciiread("matching_cloud_files_level2.txt", -1, "string")
    geoloc_files  = asciiread("matching_geoloc_files_level2.txt", -1, "string")

  ; Define directories and output location
  ;diri1 = "/work/bb1143/b381362/MODIS_data/cloud/"
  ;diri2 = "/work/bb1143/b381362/MODIS_data/geoloc/"
  ;output = "/work/bb1143/b381362/MODIS_data/"
  
  diri1 = "/scratch/b/b381362/MODIS_data/cloud/"
  diri2 = "/scratch/b/b381362/MODIS_data/geoloc/"
  output = "/scratch/b/b381362/MODIS_data/"

  ; Get the number of matching cloud files and print it
  nfils = dimsizes(cloud_files)
  print("Processing file number: " + sprinti("%d", nfils))

  ; Loop over each matching cloud file
  do i = 0, nfils - 1
      cloudfile = cloud_files(i)
      geolocfile = geoloc_files(i)
      
      print("processing: " + cloudfile + " with " + geolocfile)

      ; Extract substring from the start up to just before the ".hdf" extension
      name_file=str_split_by_length(cloudfile, 40)
      print("name_out_file: " + name_file(0) )

      ; Call your processing function (ensure processFile is defined)
      processFile_level2(diri1 + cloudfile, diri2 + geolocfile, name_file, output)
  end do
end
