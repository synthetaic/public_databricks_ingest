// Databricks notebook source
def mountFolder(dstStorageAccount:String, dstContainerName:String, dstFolderPath:String) {
  import java.lang.IllegalArgumentException
  
  val containerName     : String = dstContainerName
  val folderName        : String = dstFolderPath
  val storageAccount    : String = dstStorageAccount
  val appRegistrationId : String = "5da2fe00-7ce9-4be8-8f11-57befc2cf831"  // Application (client) ID
  val directoryId       : String = "628574bc-9c62-4216-afdd-7278cbd0123a"  // Directory (tenant) ID

  // Acquire secret from Azure Key Vault
//   val clientSecret = dbutils.secrets.get(scope = "azure-key-vault", key = "dev-raic-ingest-adlsgen2-001")
  // dev-raic--ingest-adlsgen2-001
  val clientSecret = "n9O7Q~EwVaFljyDlKV4VyXfYmIjK4qUW2osNp"
//   val clientSecret = "1xqq57wPSkk8XIvqde+wpEx+k9lvD7tytKJOuOqIPmCuiCO4eCQTygugp5EgAXZWROfL3AOsn6ACtm0JBlE57Q=="
  
  val sas_key = "?sv=2020-08-04&ss=bfqt&srt=co&sp=rwlacupitf&se=2028-02-09T05:38:27Z&st=2022-02-08T21:38:27Z&spr=https&sig=LqDunHmud2LsmoU5LmpswsE2jWrOGVn7Pzm6Ehl32OM%3D"
  
  // Uses defined properties from above
  val storageSource         : String = s"abfss://$containerName@$storageAccount.dfs.core.windows.net/$folderName"
  val fullMountLocationName : String = s"/mnt/$containerName/$folderName"
  
//   val storageSource         : String = s"wasbs://$containerName@$storageAccount.blob.core.windows.net/"
//   val fullMountLocationName : String = s"/mnt/$containerName/" 
  
  val configs = Map(
          "fs.azure.account.auth.type"              -> "OAuth"
        , "fs.azure.account.oauth.provider.type"    -> "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
        , "fs.azure.account.oauth2.client.id"       -> appRegistrationId
        , "fs.azure.account.oauth2.client.secret"   -> clientSecret
        , "fs.azure.account.oauth2.client.endpoint" -> s"https://login.microsoftonline.com/$directoryId/oauth2/token"
        )

  try {
        dbutils.fs.mount(
              source = storageSource
              , mountPoint = fullMountLocationName
              , extraConfigs = configs
//               , extraConfigs = Map(s"fs.azure.sas.$containerName.$storageAccount.blob.core.windows.net" -> sas_key)

              ) 
        println(s"Mount Location Created at: /mnt/$containerName/$folderName")
    
  } catch {
        case e: java.rmi.RemoteException => {
          println(s"Directory is Already Mounted at: /mnt/$containerName/$folderName")
        }
    
        case e: Exception => {
          println("There was some other error")
        }
  }
}

def listMounts() {
  // list out all mount locations
  display(dbutils.fs.mounts().toDF) 
}

def unMountFolder(dstFolderPath:String) {
    dbutils.fs.unmount(dstFolderPath)
}

// COMMAND ----------

// storage_account_name = "devstraicscussynth"
// container_name = "maps"
// sas_key = "?sv=2020-08-04&ss=bfqt&srt=co&sp=rwlacupitf&se=2028-02-09T05:38:27Z&st=2022-02-08T21:38:27Z&spr=https&sig=LqDunHmud2LsmoU5LmpswsE2jWrOGVn7Pzm6Ehl32OM%3D"

// source_uri = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net"
// mount_point = f"/mnt/devraic"

// sas_conf = f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net"

// mountFolder("rebellion", "dev-databricks", "input")

mountFolder("rebellion", "upload-00002", "CS-5343")
mountFolder("rebellion", "dev-databricks", "output")

// mountFolder("devstraicscussynth", "maps", "mapsraw")
// mountFolder("devstraicscussynth", "maps-databricks", "mapsdb")

// COMMAND ----------

// Simple function to list all mounted locations
listMounts()

// COMMAND ----------

// dbutils.fs.ls("dbfs:/mnt/upload-00002/CS-5343")
dbutils.fs.ls("dbfs:/mnt/dev-databricks/output")


// COMMAND ----------

unMountFolder("/mnt/dev-databricks/output")
unMountFolder("/mnt/upload-00002/CS-5343")

// COMMAND ----------


